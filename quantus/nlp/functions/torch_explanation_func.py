"""Explanation functions for Torch models."""

from __future__ import annotations
from typing import List, Callable, Optional, Dict, TYPE_CHECKING

import numpy as np
import torch

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation, ExplainFn, NumericalExplainFn
from quantus.nlp.helpers.utils import (
    unpack_token_ids_and_attention_mask,
    get_interpolated_inputs,
    value_or_default,
    map_optional,
    apply_noise,
)

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TensorLike

    BaselineFn = Callable[[TensorLike], TensorLike]


def torch_explain_gradient_norm(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
) -> List[Explanation]:

    device = model.device  # noqa

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokens)
    attention_mask = torch.tensor(attention_mask).to(device)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _torch_explain_gradient_norm(
        input_embeds,
        attention_mask,
        y_batch,
        model,
    )
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def _torch_explain_gradient_norm(
    input_embeddings: TensorLike,
    attention_mask: Optional[TensorLike],
    y_batch: np.ndarray,
    model: TextClassifier,
) -> np.ndarray:

    device = model.device  # noqa

    input_embeddings = torch.tensor(input_embeddings, requires_grad=True).to(device)
    logits = model(input_embeddings, attention_mask)
    indexes = torch.reshape(torch.tensor(y_batch).to(device), (len(y_batch), 1))
    logits_for_class = torch.gather(logits, dim=-1, index=indexes)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.linalg.norm(grads, dim=-1).detach().cpu().numpy()


def torch_explain_input_x_gradient(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
) -> List[Explanation]:
    device = model.device  # noqa
    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokens)
    attention_mask = torch.tensor(attention_mask).to(device)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _torch_explain_input_x_gradient(
        input_embeds,
        attention_mask,
        y_batch,
        model,
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def _torch_explain_input_x_gradient(
    input_embeddings: TensorLike,
    attention_mask: Optional[TensorLike],
    y_batch: np.ndarray,
    model: TextClassifier,
) -> np.ndarray:

    device = model.device  # noqa

    input_embeddings = torch.tensor(input_embeddings, requires_grad=True).to(device)
    logits = model(input_embeddings, attention_mask)
    indexes = torch.reshape(torch.tensor(y_batch).to(device), (len(y_batch), 1))
    logits_for_class = torch.gather(logits, dim=-1, index=indexes)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach().cpu().numpy()


def torch_explain_integrated_gradients(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[BaselineFn] = None,
) -> List[Explanation]:

    device = model.device  # noqa

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = unpack_token_ids_and_attention_mask(tokens)

    input_embeds = model.embedding_lookup(input_ids)

    scores = _torch_explain_integrated_gradients(
        input_embeds,
        attention_mask,
        y_batch,
        model,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def _torch_explain_integrated_gradients(
    input_embeddings: TensorLike,
    attention_mask: Optional[TensorLike],
    y_batch: np.ndarray,
    model: TextClassifier,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[BaselineFn] = None,
) -> np.ndarray:
    device = model.device  # noqa

    baseline_fn = value_or_default(baseline_fn, lambda: lambda x: np.zeros_like(x))

    interpolated_embeddings = []
    interpolated_attention_mask = None if attention_mask is None else []

    for i, embeddings_i in enumerate(input_embeddings):
        np_embeddings_i = embeddings_i.detach().cpu()
        interpolated_embeddings.append(
            get_interpolated_inputs(
                baseline_fn(np_embeddings_i), np_embeddings_i, num_steps
            )
        )
        if attention_mask is not None:

            interpolated_attention_mask.append(
                np.broadcast_to(
                    attention_mask[i], (num_steps + 1, *attention_mask[i].shape)
                )
            )

    interpolated_embeddings = torch.tensor(
        interpolated_embeddings, dtype=torch.float32, requires_grad=True
    ).to(device)
    interpolated_embeddings = torch.reshape(
        interpolated_embeddings, [-1, *interpolated_embeddings.shape[2:]]
    )

    if interpolated_attention_mask is not None:
        interpolated_attention_mask = torch.tensor(interpolated_attention_mask).to(
            device
        )
        interpolated_attention_mask = torch.reshape(
            interpolated_attention_mask, [-1, *interpolated_attention_mask.shape[2:]]
        )

    logits = model(interpolated_embeddings, interpolated_attention_mask)
    indexes = torch.reshape(torch.tensor(y_batch).to(device), (len(y_batch), 1))
    logits_for_class = torch.gather(logits, dim=-1, index=indexes)
    grads = torch.autograd.grad(
        torch.unbind(logits_for_class), interpolated_embeddings
    )[0]
    grads = torch.reshape(
        grads, [len(input_embeddings), num_steps + 1, *grads.shape[1:]]
    )
    scores = torch.trapz(torch.trapz(grads, dim=-1), dim=1)
    return scores.detach().cpu()


_noise_grad_explain_fn_map = {
    "GradNorm": _torch_explain_gradient_norm,
    "InputXGrad": _torch_explain_input_x_gradient,
    "IntGrad": _torch_explain_integrated_gradients,
}


def torch_explain_noise_grad_plus_plus(
    model: TextClassifier, x_batch: List[str], y_batch: np.ndarray, *args, **kwargs
) -> List[Explanation]:
    pass


_plain_text_method_mapping: Dict[str, ExplainFn] = {
    "GradNorm": torch_explain_gradient_norm,
    "InputXGrad": torch_explain_input_x_gradient,
    "IntGrad": torch_explain_integrated_gradients,
    "NoiseGrad++": torch_explain_noise_grad_plus_plus,
}

_numerical_method_mapping: Dict[str, NumericalExplainFn] = {}


def torch_explain(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
    method: str,
    **kwargs,
) -> List[Explanation]:
    pass
