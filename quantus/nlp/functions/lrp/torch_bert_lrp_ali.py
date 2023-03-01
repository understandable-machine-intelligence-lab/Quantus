"""Based on https://github.com/AmeenAli/XAI_Transformers."""

from __future__ import annotations
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from torch import nn
import torch
import copy
from dataclasses import dataclass
from typing import Optional, List, Tuple
from torch import Tensor, device

from quantus.nlp.helpers.utils import map_optional


@dataclass
class BertLRPConfig:
    hidden_size: int
    num_attention_heads: int
    layer_norm_eps: float
    num_classes: int
    num_attention_blocks: int
    attention_head_size: int
    all_head_size: int
    detach_layernorm: bool  # Detaches the attention-block-output LayerNorm
    detach_kq: bool  # Detaches the kq-softmax branch
    detach_mean: bool
    device: device


@dataclass
class LayerNormArgs:
    mean_detach: bool
    std_detach: bool


class BertForSequenceClassificationLRP(nn.Module):
    def __init__(self, config: BertLRPConfig, embeddings: BertEmbeddings):
        super().__init__()

        n_blocks = config.num_attention_blocks
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(
            *[BertSelfAttentionLRP(config) for _ in range(n_blocks)]
        )
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(
            in_features=config.hidden_size, out_features=config.num_classes, bias=True
        )
        self.device = config.device

    def explain(
        self,
        input_ids: Optional[Tensor],
        y_batch: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        gammas: float | List[float] = 0.01,
    ) -> Tensor:
        hidden_states = self.embeds(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        extended_attention_mask = map_optional(
            attention_mask,
            lambda x: get_extended_attention_mask(x, input_shape).to(self.device),
        )

        attn_input = hidden_states

        attention_inputs = []
        attention_inputs_data = []

        for i, attention_block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            attention_inputs_data.append(attn_inputdata)
            attention_inputs.append(attn_input)

            gamma = gammas if isinstance(gammas, float) else gammas[i]
            output = attention_block(
                attn_inputdata, gamma=gamma, attention_mask=extended_attention_mask
            )
            attn_input = output

        outputdata = output.data  # noqa
        outputdata.requires_grad_(True)

        pooled = self.pooler(outputdata)

        pooleddata = pooled.data
        pooleddata.requires_grad_(True)

        logits = self.classifier(pooleddata)

        indexes = torch.reshape(y_batch, (len(y_batch), 1))
        logits_for_class = torch.gather(logits, dim=-1, index=indexes)

        Rout = logits_for_class

        torch.autograd.backward(torch.unbind(Rout))
        (pooleddata.grad * pooled).sum().backward()

        Rpool = outputdata.grad * output

        R_ = Rpool
        for i, attention_block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = attention_inputs_data[i].grad
            R_attn = R_grad * attention_inputs[i]
            R_ = R_attn

        R = R_.sum(2).detach()

        return R


class BertSelfAttentionLRP(nn.Module):
    def __init__(self, config: BertLRPConfig):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.output = BertSelfOutputLRP(config)
        self.detach = config.detach_kq

    def transpose_for_scores(self, x: Tensor) -> Tensor:
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        gamma: float = 0,
    ) -> Tensor:
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

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

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

        self.weight = nn.Parameter(Tensor(hidden))
        self.bias = nn.Parameter(Tensor(hidden))
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
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

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

    def player_forward(
        self, hidden_states: Tensor, input_tensor: Tensor, gamma: float
    ) -> Tensor:
        player_dense = make_player_layer(self.dense, gamma)
        hidden_states = player_dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


def make_player_layer(layer: nn.Module, gamma: float) -> nn.Module:
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight + gamma * layer.weight.clamp(min=0))
    player.bias = torch.nn.Parameter(layer.bias + gamma * layer.bias.clamp(min=0))
    return player


def get_extended_attention_mask(
    attention_mask: Tensor,
    input_shape: Tuple[int],
) -> Tensor:
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = (
            ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask
            )
        )
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
        torch.float64
    ).min
    return extended_attention_mask
