"""Based on https://github.com/AmeenAli/XAI_Transformers."""

from __future__ import annotations
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from torch import nn
import torch
import copy
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BertLRPConfig:
    hidden_size: int
    num_attention_heads: int
    layer_norm_eps: float
    n_classes: int
    n_blocks: int
    attention_head_size: int
    all_head_size: int
    detach_layernorm: bool  # Detaches the attention-block-output LayerNorm
    detach_kq: bool  # Detaches the kq-softmax branch
    detach_mean: bool
    device: str


@dataclass
class LayerNormArgs:
    mean_detach: bool
    std_detach: bool


class BertForSequenceClassificationLRP(nn.Module):
    def __init__(self, config: BertLRPConfig, embeddings: BertEmbeddings):
        super().__init__()

        n_blocks = config.n_blocks
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(
            *[BertSelfAttentionLRP(config) for _ in range(n_blocks)]
        )
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(
            in_features=config.hidden_size, out_features=config.n_classes, bias=True
        )
        self.device = config.device

    def forward_and_explain(
        self,
        input_ids: Optional[torch.Tensor],
        y_batch: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
        gammas: float | List[float] = 0.01,
    ):
        # Forward
        A = {}

        hidden_states = self.embeds(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        ).to(self.config.device)

        A["hidden_states"] = hidden_states

        attn_input = hidden_states

        for i, attention_block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            A["attn_input_{}_data".format(i)] = attn_inputdata
            A["attn_input_{}".format(i)] = attn_input

            gamma = gammas if isinstance(gammas, float) else gammas[i]

            output, attention_probs = attention_block(
                A["attn_input_{}_data".format(i)], gamma=gamma
            )
            attn_input = output

        # (1, 12, 768) -> (1x768)

        outputdata = output.data
        outputdata.requires_grad_(True)

        pooled = self.pooler(outputdata)  # A['attn_output'] )

        # (1x768) -> (1,nclasses)
        pooleddata = pooled.data
        pooleddata.requires_grad_(True)

        logits = self.classifier(pooleddata)

        A["logits"] = logits

        # Through clf layer
        Rout = A["logits"][:, y_batch]

        self.R0 = Rout.detach().cpu().numpy()

        Rout.backward()
        ((pooleddata.grad) * pooled).sum().backward()

        Rpool = (outputdata.grad) * output

        R_ = Rpool
        for i, attention_block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = A["attn_input_{}_data".format(i)].grad
            R_attn = (R_grad) * A["attn_input_{}".format(i)]
            R_ = R_attn

        R = R_.sum(2).detach().cpu().numpy()

        return {"logits": logits, "R": R}


class BertSelfAttentionLRP(nn.Module):
    def __init__(self, config: BertLRPConfig):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.output = BertSelfOutputLRP(config)
        self.detach = config.detach_kq

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])
        new_x_shape = x.size()[:-1] + (
            self.config.num_attention_heads,
            self.config.attention_head_size,
        )
        x = x.view(*new_x_shape)
        x_out = x.permute(0, 2, 1, 3)
        return x_out

    @staticmethod
    def pproc(layer: nn.Module, player, x):
        # ??? what does pproc mean ???
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data

    def forward(self, hidden_states: torch.Tensor, gamma: float = 0) -> torch.Tensor:
        pquery = make_player_layer(self.query, gamma)
        pkey = make_player_layer(self.key, gamma)
        pvalue = make_player_layer(self.value, gamma)

        query_ = self.pproc(self.query, pquery, hidden_states)
        key_ = self.pproc(self.key, pkey, hidden_states)
        val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, seq len, 768] -> [1, 12, seq len, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([1, 12, 10, 10])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))

        if self.detach:
            attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.config.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.output.player_forward(context_layer, hidden_states, gamma=gamma)
        return output


class LayerNormImpl(nn.Module):
    __constants__ = ["weight", "bias", "eps"]

    def __init__(
        self,
        args: LayerNormArgs,
        hidden: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super(LayerNormImpl, self).__init__()
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = nn.Parameter(torch.Tensor(hidden))
        self.bias = nn.Parameter(torch.Tensor(hidden))
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        std = inputs.std(dim=-1, keepdim=True)
        if self.mean_detach:
            mean = mean.detach()
        if self.std_detach:
            std = std.detach()
        input_norm = (inputs - mean) / (std + self.eps)
        return input_norm


class BertSelfOutputLRP(nn.Module):
    def __init__(self, config: BertLRPConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

        layer_norm_args = LayerNormArgs(
            mean_detach=config.detach_mean, std_detach=config.detach_layernorm
        )
        self.layer_norm = LayerNormImpl(
            layer_norm_args, config.hidden_size, eps=config.layer_norm_eps
        )

        self.detach = config.detach_layernorm

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

    def player_forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        player_dense = make_player_layer(self.dense, gamma)
        hidden_states = player_dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


def make_player_layer(layer: nn.Module, gamma: float) -> nn.Module:
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight + gamma * layer.weight.clamp(min=0))
    player.bias = torch.nn.Parameter(layer.bias + gamma * layer.bias.clamp(min=0))
    return player
