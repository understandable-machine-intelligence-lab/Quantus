"""Based on https://github.com/AmeenAli/XAI_Transformers."""

from torch import nn
import torch
import copy
import torch.functional as F


class LayerNormImpl(nn.Module):
    __constants__ = ["weight", "bias", "eps"]

    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == "no_norm":
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == "no_norm":
            return input
        elif self.mode == "topk":
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps
            )
        elif self.mode == "adanorm":
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm * self.adanorm_scale
        elif self.mode == "nowb":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == "distillnorm":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)

            input_norm = input_norm * self.weight + self.bias

            return input_norm

        elif self.mode == "gradnorm":
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output


def LayerNorm(
    normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None
):
    if args is not None:
        if args.lnv != "origin":
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LNargs(object):
    def __init__(self):
        self.lnv = "nowb"
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = False


class LNargsDetach(object):
    def __init__(self):
        self.lnv = "nowb"
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = True
        self.std_detach = True


class LNargsDetachNotMean(object):
    def __init__(self):
        self.lnv = "nowb"
        self.sigma = None
        self.hidden = None
        self.adanorm_scale = 1.0
        self.nowb_scale = None
        self.mean_detach = False
        self.std_detach = True


def make_p_layer(layer, gamma):
    player = copy.deepcopy(layer)
    player.weight = torch.nn.Parameter(layer.weight + gamma * layer.weight.clamp(min=0))
    player.bias = torch.nn.Parameter(layer.bias + gamma * layer.bias.clamp(min=0))
    return player


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        self.first_token_tensor = hidden_states[:, 0]
        self.pooled_output1 = self.dense(self.first_token_tensor)
        self.pooled_output2 = self.activation(self.pooled_output1)
        return self.pooled_output2


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

        if self.config.train_mode == True:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        if config.detach_layernorm == True:
            assert config.train_mode == False

            if config.detach_mean == False:
                print("Detach LayerNorm only Norm")
                largs = LNargsDetachNotMean()
            else:
                print("Detach LayerNorm Mean+Norm")
                largs = LNargsDetach()
        else:
            largs = LNargs()

        self.LayerNorm = LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, args=largs
        )

        self.detach = config.detach_layernorm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if self.config.train_mode == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def pforward(self, hidden_states, input_tensor, gamma):
        pdense = make_p_layer(self.dense, gamma)
        hidden_states = pdense(hidden_states)
        # hidden_states = self.dense(hidden_states)
        if self.config.train_mode == True:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.all_head_size)
        self.key = nn.Linear(config.hidden_size, config.all_head_size)
        self.value = nn.Linear(config.hidden_size, config.all_head_size)
        self.output = BertSelfOutput(config)
        self.detach = config.detach_kq
        if self.config.train_mode == True:
            self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        if self.detach == True:
            assert self.config.train_mode == False
            print("Detach K-Q-softmax branch")

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def transpose_for_scores(self, x):
        # x torch.Size([1, 10, 768])
        # xout torch.Size([1, 10, 12, 64])
        new_x_shape = x.size()[:-1] + (
            self.config.num_attention_heads,
            self.config.attention_head_size,
        )
        x = x.view(*new_x_shape)
        X = x.permute(0, 2, 1, 3)
        return X

    def un_transpose_for_scores(self, x, old_shape):
        x = x.permute(0, 1, 2, 3)
        return x.reshape(old_shape)

    @staticmethod
    def pproc(layer, player, x):
        z = layer(x)
        zp = player(x)
        return zp * (z / zp).data

    def forward(self, hidden_states, gamma=0, method=None):
        #  print('PKQ gamma', gamma)

        pquery = make_p_layer(self.query, gamma)
        pkey = make_p_layer(self.key, gamma)
        pvalue = make_p_layer(self.value, gamma)

        n_nodes = hidden_states.shape[1]

        if self.config.train_mode:
            query_ = self.query(hidden_states)
            key_ = self.key(hidden_states)
            val_ = self.value(hidden_states)
        else:
            query_ = self.pproc(self.query, pquery, hidden_states)
            key_ = self.pproc(self.key, pkey, hidden_states)
            val_ = self.pproc(self.value, pvalue, hidden_states)

        # [1, senlen, 768] -> [1, 12, senlen, 64]
        query_t = self.transpose_for_scores(query_)
        key_t = self.transpose_for_scores(key_)
        val_t = self.transpose_for_scores(val_)

        # torch.Size([1, 12, 10, 64]) , torch.Size([1, 12, 64, 10]) -> torch.Size([1, 12, 10, 10])
        attention_scores = torch.matmul(query_t, key_t.transpose(-1, -2))

        # if torch.isnan(attention_scores).any():
        #    import pdb;pdb.set_trace()

        if self.detach:
            assert self.config.train_mode == False

            attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.config.train_mode:
            attention_probs = self.dropout(attention_probs)
        if method == "GAE":
            attention_probs.register_hook(self.save_attn_gradients)

        context_layer = torch.matmul(attention_probs, val_t)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        old_context_layer_shape = context_layer.shape
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.config.all_head_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.config.train_mode:
            output = self.output(context_layer, hidden_states)
        else:
            #
            #  print('Out gamma', gamma)
            output = self.output.pforward(context_layer, hidden_states, gamma=gamma)

        return (
            output,
            attention_probs,
        )  # , (attention_scores, hidden_states) #, query_t, key_t, val_t)


class BertAttention(nn.Module):
    def __init__(self, config, embeddings):
        super().__init__()

        n_blocks = config.n_blocks
        self.n_blocks = n_blocks
        self.embeds = embeddings

        self.config = config
        self.attention_layers = torch.nn.Sequential(
            *[AttentionBlock(config) for i in range(n_blocks)]
        )
        self.output = BertSelfOutput(config)

        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(
            in_features=config.hidden_size, out_features=config.n_classes, bias=True
        )
        self.device = config.device

        self.attention_probs = {i: [] for i in range(n_blocks)}
        self.attention_debug = {i: [] for i in range(n_blocks)}
        self.attention_gradients = {i: [] for i in range(n_blocks)}
        self.attention_cams = {i: [] for i in range(n_blocks)}

        self.attention_lrp_gradients = {i: [] for i in range(n_blocks)}

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
    ):
        hidden_states = self.embeds(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        ).to(self.config.device)

        attn_input = hidden_states
        for i, block in enumerate(self.attention_layers):
            output, attention_probs = block(attn_input)

            self.attention_probs[i] = attention_probs
            #  self.attention_debug[i] = debug_data +  (output,)
            attn_input = output

        pooled = self.pooler(output)
        logits = self.classifier(pooled)

        self.output_ = output
        self.pooled_ = pooled
        self.logits_ = logits

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {"loss": loss, "logits": logits}

    def prep_lrp(self, x):
        x = x.data
        x.requires_grad_(True)
        return x

    def forward_and_explain(
        self,
        input_ids,
        cl,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        past_key_values_length=0,
        gammas=None,
        method=None,
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

        for i, block in enumerate(self.attention_layers):
            # [1, 12, 768] -> [1, 12, 768]
            attn_inputdata = attn_input.data
            attn_inputdata.requires_grad_(True)

            A["attn_input_{}_data".format(i)] = attn_inputdata
            A["attn_input_{}".format(i)] = attn_input

            gamma = 0.0 if gammas is None else gammas[i]
            #  print('using gamma', gamma)

            output, attention_probs = block(
                A["attn_input_{}_data".format(i)], gamma=gamma, method=method
            )

            self.attention_probs[i] = attention_probs
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
        Rout = A["logits"][:, cl]

        self.R0 = Rout.detach().cpu().numpy()

        Rout.backward()
        ((pooleddata.grad) * pooled).sum().backward()

        Rpool = (outputdata.grad) * output

        R_ = Rpool
        for i, block in list(enumerate(self.attention_layers))[::-1]:
            R_.sum().backward()

            R_grad = A["attn_input_{}_data".format(i)].grad
            R_attn = (R_grad) * A["attn_input_{}".format(i)]
            if method == "GAE":
                self.attention_gradients[i] = block.get_attn_gradients().squeeze()
            R_ = R_attn

        R = R_.sum(2).detach().cpu().numpy()

        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
        else:
            loss = None

        return {"loss": loss, "logits": logits, "R": R}
