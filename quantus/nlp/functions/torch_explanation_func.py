# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Explanation functions for Torch models."""

from __future__ import annotations

from functools import singledispatch
from typing import Callable, Dict, List, Optional, Union
from operator import itemgetter

import numpy as np
import torch
from torch import Tensor

from quantus.helpers.utils import map_dict
from quantus.nlp.helpers.model.torch_model import TorchHuggingFaceTextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import (
    get_input_ids,
    value_or_default,
    get_scores,
)

# Just to save some typing effort
_TextOrVector = Union[List[str], Tensor, np.ndarray]
_TextOrTensor = Union[List[str], Tensor]
_Scores = Union[List[Explanation], np.ndarray]


# ----------------- "Entrypoint" --------------------


def available_xai_methods() -> Dict:
    return {
        "GradNorm": gradient_norm,
        "GradXInput": gradient_x_input,
        "IntGrad": integrated_gradients,
        "NoiseGrad++": noise_grad_plus_plus,
        "NoiseGrad": noise_grad,
    }


def torch_explain(
    model: TorchHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    *args,
    method: str,
    **kwargs,
) -> _Scores:
    method_mapping = available_xai_methods()

    if method not in method_mapping:
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(method_mapping.keys())}"
        )
    explain_fn = method_mapping[method]

    if isinstance(x_batch, np.ndarray):
        x_batch = model.to_tensor(x_batch, requires_grad=True)

    y_batch = model.to_tensor(y_batch, dtype=torch.int64)

    return explain_fn(model, x_batch, y_batch, *args, **kwargs)


# ----------------- Quantus-conform API -------------------
# functools.singledispatch supports only dispatching based on 1st argument type,
# which in our case is model, so we need to reorder them, so x_batch (text or embedding) is in 1st place,
# and we and up dispatching to different functions based on input type.


def gradient_norm(
    model: TorchHuggingFaceTextClassifier,
    x_batch: _TextOrTensor,
    y_batch: Tensor,
    **kwargs,
) -> _Scores:
    """
    A baseline GradientNorm text-classification explainer. GradientNorm explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute L2 norm of gradients.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _gradient_norm(x_batch, model, y_batch, **kwargs)


def gradient_x_input(
    model: TorchHuggingFaceTextClassifier,
    x_batch: _TextOrTensor,
    y_batch: Tensor,
    **kwargs,
) -> _Scores:
    """
    A baseline GradientXInput text-classification explainer.GradientXInput explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute vector dot product between input embeddings and gradients.


    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _gradient_x_input(x_batch, model, y_batch, **kwargs)


def integrated_gradients(
    model: TorchHuggingFaceTextClassifier,
    x_batch: List[str],
    y_batch: Tensor,
    num_steps: int = 10,
) -> _Scores:
    """
    This function depends on transformers_interpret library.
    A baseline Integrated Gradients text-classification explainer. Integrated Gradients explanation algorithm is:
        - Convert inputs to models latent representations.
        - For each x, y in x_batch, y_batch
        - Generate num_steps samples interpolated from baseline to x.
        - Execute forwards pass.
        - Retrieve logits for y.
        - Compute gradient of logits with respect to interpolated samples.
        - Estimate integral over interpolated samples using trapezoid rule.
    In practise, we combine all interpolated samples in one batch, to avoid executing forward and backward passes
    in for-loop. This means potentially, that batch size selected for this XAI method should be smaller than usual.

    References:
    ----------
    - Sundararajan et al., 2017, Axiomatic Attribution for Deep Networks, https://arxiv.org/pdf/1703.01365.pdf
    - https://github.com/cdpierse/transformers-interpret

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    num_steps:
        Number of interpolated samples, which should be generated, default=10.
    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    """

    return _integrated_gradients(
        x_batch,
        model,
        y_batch,
        num_steps=num_steps,
    )


def noise_grad(
    model: TorchHuggingFaceTextClassifier,
    x_batch: _TextOrTensor,
    y_batch: Tensor,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> _Scores:
    """
    NoiseGrag is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's. This method requires noisegrad package,
    install it with: `pip install 'noisegrad @ git+https://github.com/aaarrti/NoiseGrad.git'`.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad.
        Otherwise, must have `Callable[[HuggingFaceTextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    init_kwargs:
        Kwargs passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    init_kwargs = value_or_default(init_kwargs, lambda: {})
    return _noise_grad(
        x_batch,
        model,
        y_batch,
        explain_fn=explain_fn,
        init_kwargs=init_kwargs,
        **kwargs,
    )


def noise_grad_plus_plus(
    model: TorchHuggingFaceTextClassifier,
    x_batch: _TextOrTensor,
    y_batch: Tensor,
    *,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> _Scores:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's weights and model's inputs. This method requires noisegrad package,
    install it with: `pip install 'noisegrad @ git+https://github.com/aaarrti/NoiseGrad.git'`.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad.
        Otherwise, must have `Callable[[HuggingFaceTextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    init_kwargs:
        Kwargs passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    init_kwargs = value_or_default(init_kwargs, lambda: {})
    return _noise_grad_plus_plus(
        x_batch,
        model,
        y_batch,
        explain_fn=explain_fn,
        init_kwargs=init_kwargs,
        **kwargs,
    )


# ------------ single dispatch stubs  --------------


@singledispatch
def _gradient_norm(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _gradient_x_input(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _integrated_gradients(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _noise_grad(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _noise_grad_plus_plus(x_batch, *args, **kwargs) -> _Scores:
    pass


# ----------------------- GradNorm -------------------------


@_gradient_norm.register
def _(
    x_batch: list, model: TorchHuggingFaceTextClassifier, y_batch: Tensor
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _gradient_norm(input_embeds, model, y_batch, **kwargs)
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_gradient_norm.register
def _(
    x_batch: Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    **kwargs,
) -> np.ndarray:
    input_embeddings = x_batch.to(model.unwrap().dtype)

    kwargs = map_dict(kwargs, model.to_tensor)

    logits = model(input_embeddings, **kwargs)

    logits_for_class = _logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.linalg.norm(grads, dim=-1).detach().cpu().numpy()


# ----------------------- GradXInput -------------------------


@_gradient_x_input.register
def _(
    x_batch: list, model: TorchHuggingFaceTextClassifier, y_batch: Tensor
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _gradient_x_input(input_embeds, model, y_batch)

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_gradient_x_input.register
def _(
    x_batch: Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    **kwargs,
) -> np.ndarray:
    input_embeddings = x_batch.to(model.unwrap().dtype)
    kwargs = map_dict(kwargs, model.to_tensor)

    logits = model(input_embeddings, **kwargs)
    logits_for_class = _logits_for_labels(logits, model.to_tensor(y_batch))
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach().cpu().numpy()


# ----------------------- IntGrad -------------------------


@_integrated_gradients.register
def _(
    x_batch: list,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    num_steps: int,
) -> List[Explanation]:
    from quantus.nlp.functions.transorfmers_interpret_adapter import IntGradAdapter

    return IntGradAdapter.explain_batch(model, x_batch, y_batch, num_steps=num_steps)


@_integrated_gradients.register
def _(
    x_batch: Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    num_steps: int,
) -> np.ndarray:
    from quantus.nlp.functions.transorfmers_interpret_adapter import (
        IntGradLatentAdapter,
    )

    return IntGradLatentAdapter.explain_batch(
        model, x_batch, y_batch, num_steps=num_steps
    )


# ----------------------- NoiseGrad -------------------------


@_noise_grad.register
def _(
    x_batch: list,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    explain_fn: Union[Callable, str],
    init_kwargs: Dict,
) -> List[Explanation]:
    from noisegrad import NoiseGrad

    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    baseline_tokens = explain_fn(model, x_batch, y_batch)  # type: ignore
    baseline_tokens = list(map(itemgetter(0), baseline_tokens))

    og_weights = model.weights.copy()

    def adapter(module, inputs, targets):
        model.weights = module.state_dict()
        a_batch = explain_fn(model, x_batch, targets)
        base_scores = get_scores(a_batch)
        return model.to_tensor(base_scores)

    ng = NoiseGrad(**init_kwargs)
    scores = (
        ng.enhance_explanation(
            model.unwrap(),
            x_batch,  # noqa
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    model.weights = og_weights
    return list(zip(baseline_tokens, scores))


@_noise_grad.register
def _(
    x_batch: Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    explain_fn: Callable,
    init_kwargs: Dict,
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGrad

    og_weights = model.weights.copy()

    def adapter(module, inputs, targets):
        model.weights = module.state_dict()
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return model.to_tensor(base_scores)

    ng_pp = NoiseGrad(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.unwrap(),
            x_batch,
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    model.weights = og_weights
    return scores


# ----------------------- NoiseGrad++ -------------------------


@_noise_grad_plus_plus.register
def _(
    x_batch: list,
    model: TorchHuggingFaceTextClassifier,
    y_batch: torch.Tensor,
    explain_fn: Union[Callable, str],
    init_kwargs: Dict,
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)

    scores = _noise_grad_plus_plus(
        input_embeds, model, y_batch, explain_fn, init_kwargs, **kwargs
    )

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_noise_grad_plus_plus.register
def _(
    x_batch: torch.Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: torch.Tensor,
    explain_fn: Callable,
    init_kwargs: Optional[Dict],
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGradPlusPlus

    og_weights = model.weights.copy()
    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    def adapter(module, inputs, targets):
        model.weights = module.state_dict()
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return model.to_tensor(base_scores)

    ng_pp = NoiseGradPlusPlus(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.unwrap(),
            x_batch,
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    model.weights = og_weights
    return scores


# -------------- utils ------------------


def _get_noise_grad_baseline_explain_fn(explain_fn: Callable | str):
    if isinstance(explain_fn, Callable):
        return explain_fn

    if explain_fn in ("NoiseGrad", "NoiseGrad++"):
        raise ValueError(f"Can't use {explain_fn} as baseline function for NoiseGrad.")
    method_mapping = available_xai_methods()
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]


def _logits_for_labels(logits: Tensor, y_batch: Tensor) -> Tensor:
    return logits[torch.range(0, logits.shape[0] - 1, dtype=torch.int64), y_batch]
