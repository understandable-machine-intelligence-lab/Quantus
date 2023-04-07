# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Explanation functions for Torch models."""

from __future__ import annotations

from functools import singledispatch, partial
from operator import itemgetter
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from torch import Tensor

from quantus.helpers.types import Explanation
from quantus.helpers.utils import map_dict
from quantus.helpers.utils import value_or_default

# Just to save some typing effort
_TextOrVector = Union[List[str], Tensor, np.ndarray]
_TextOrTensor = Union[List[str], Tensor]
_Scores = Union[List[Explanation], np.ndarray]

# ----------------- "Entrypoint" --------------------


class Config:
    return_np_arrays: bool = True

    def enable_numpy_conversion(self):
        self.return_np_arrays = True

    def disable_numpy_conversion(self):
        self.return_np_arrays = False


config = Config()


def available_xai_methods() -> Dict:
    return {
        "GradNorm": gradient_norm,
        "GradXInput": gradient_x_input,
        "IntGrad": integrated_gradients,
        "NoiseGrad++": noise_grad_plus_plus,
        "NoiseGrad": noise_grad,
    }


def torch_explain(
    model,
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
        x_batch = model.to_tensor(x_batch, requires_grad=True, dtype=model.input_dtype)

    y_batch = model.to_tensor(y_batch, dtype=torch.int64)

    return explain_fn(model, x_batch, y_batch, *args, **kwargs)


# ----------------- Quantus-conform API -------------------
# functools.singledispatch supports only dispatching based on 1st argument type,
# which in our case is model, so we need to reorder them, so x_batch (text or embedding) is in 1st place,
# and we and up dispatching to different functions based on input type.


def gradient_norm(
    model,
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
    model,
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
    model,
    x_batch: List[str],
    y_batch: Tensor,
    num_steps: int = 10,
    **kwargs,
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

    return _integrated_gradients(x_batch, model, y_batch, num_steps=num_steps, **kwargs)


def noise_grad(
    model,
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
    model,
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


# ----------------------- GradNorm -------------------------


@singledispatch
def _gradient_norm(
    x_batch: list, model, y_batch: Tensor, **kwargs
) -> List[Explanation]:
    input_ids, kwargs = model.tokenizer.get_input_ids(x_batch)
    input_ids = model.to_tensor(input_ids)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _gradient_norm(input_embeds, model, y_batch, **kwargs)
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


@_gradient_norm.register
def _(
    x_batch: Tensor,
    model,
    y_batch: Tensor,
    **kwargs,
) -> np.ndarray:
    logits = model(x_batch, **kwargs)
    logits_for_class = _logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), x_batch)[0]

    scores = torch.linalg.norm(grads, dim=-1)
    if config.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


# ----------------------- GradXInput -------------------------


@singledispatch
def _gradient_x_input(
    x_batch: list, model, y_batch: Tensor, **kwargs
) -> List[Explanation]:
    input_ids, kwargs = model.tokenizer.get_input_ids(x_batch)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _gradient_x_input(input_embeds, model, y_batch, **kwargs)
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


@_gradient_x_input.register
def _(
    x_batch: Tensor,
    model,
    y_batch: Tensor,
    **kwargs,
) -> np.ndarray:

    logits = model(x_batch, **kwargs)
    logits_for_class = _logits_for_labels(logits, model.to_tensor(y_batch))
    grads = torch.autograd.grad(torch.unbind(logits_for_class), x_batch)[0]
    scores = torch.sum(grads * x_batch, dim=-1)
    if config.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


# ----------------------- IntGrad -------------------------


@singledispatch
def _integrated_gradients(
    x_batch: list,
    model,
    y_batch: Tensor,
    num_steps: int,
) -> List[Explanation]:
    input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)
    x_embeddings = model.embedding_lookup(input_ids).to(model.input_dtype)
    scores = _integrated_gradients(
        x_embeddings, model, y_batch, num_steps, **predict_kwargs
    )
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


@_integrated_gradients.register
def _(
    x_batch: Tensor,
    model,
    y_batch: Tensor,
    num_steps: int,
    **kwargs,
) -> np.ndarray:
    def pseudo_interpolate(x):
        if isinstance(x, np.ndarray):
            x = model.to_tensor(x)
        if not isinstance(x, torch.Tensor):
            return x
        old_shape = x.shape
        return torch.reshape(
            torch.broadcast_to(x, (num_steps, *old_shape)),
            (old_shape[0] * num_steps, old_shape[1]),
        )

    kwargs = map_dict(kwargs, pseudo_interpolate)

    explainer = IntegratedGradients(partial(model.__call__, **kwargs))
    if model.device.type == "mps":
        step_sizes_func, alphas_func = approximation_parameters("gausslegendre")
        step_sizes, alphas = step_sizes_func(num_steps), alphas_func(num_steps)
        step_sizes = np.asarray(step_sizes).astype(np.float32)
        alphas = np.asarray(alphas).astype(np.float32)

        explainer._attribute = partial(explainer._attribute, step_sizes_and_alphas=(step_sizes, alphas))
    grads = explainer.attribute(
        inputs=x_batch,
        n_steps=num_steps,
        target=y_batch
    )

    scores = torch.linalg.norm(grads, dim=-1)

    if config.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


# ----------------------- NoiseGrad -------------------------


@singledispatch
def _noise_grad(
    x_batch: list,
    model,
    y_batch: Tensor,
    explain_fn: Union[Callable, str],
    init_kwargs: Dict,
) -> List[Explanation]:
    from noisegrad import NoiseGrad

    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    baseline_tokens = explain_fn(model, x_batch, y_batch)  # type: ignore
    baseline_tokens = list(map(itemgetter(0), baseline_tokens))

    og_weights = model.state_dict().copy()
    config.disable_numpy_conversion()

    def adapter(module, inputs, targets) -> torch.Tensor:
        model.load_state_dict(module.state_dict())
        a_batch = explain_fn(model, x_batch, targets)
        base_scores = torch.stack([i[1] for i in a_batch])
        return base_scores  # noqa

    ng = NoiseGrad(**init_kwargs)
    scores = (
        ng.enhance_explanation(
            model.get_model(),
            x_batch,  # noqa
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    config.enable_numpy_conversion()
    model.load_state_dict(og_weights)
    return list(zip(baseline_tokens, scores))


@_noise_grad.register
def _(
    x_batch: Tensor,
    model,
    y_batch: Tensor,
    explain_fn: Callable,
    init_kwargs: Dict,
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGrad

    og_weights = model.get_state_dict().copy()
    config.disable_numpy_conversion()

    def adapter(module, inputs, targets):
        model.load_state_dict(module.state_dict())
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return base_scores

    ng_pp = NoiseGrad(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.get_model(),
            x_batch,
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    config.enable_numpy_conversion()
    model.load_state_dict(og_weights)
    return scores


# ----------------------- NoiseGrad++ -------------------------


@singledispatch
def _noise_grad_plus_plus(
    x_batch: list,
    model,
    y_batch: torch.Tensor,
    explain_fn: Union[Callable, str],
    init_kwargs: Dict,
    **kwargs,
) -> List[Explanation]:
    input_ids, kwargs = model.tokenizer.get_input_ids(x_batch)
    input_embeds = model.embedding_lookup(input_ids)

    scores = _noise_grad_plus_plus(
        input_embeds, model, y_batch, explain_fn, init_kwargs, **kwargs
    )

    return [(model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_noise_grad_plus_plus.register
def _(
    x_batch: torch.Tensor,
    model,
    y_batch: torch.Tensor,
    explain_fn: Callable,
    init_kwargs: Optional[Dict],
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGradPlusPlus

    og_weights = model.state_dict().copy()
    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)
    config.disable_numpy_conversion()

    def adapter(module, inputs, targets):
        model.load_state_dict(module.state_dict())
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return base_scores

    ng_pp = NoiseGradPlusPlus(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.get_model(),
            x_batch,
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )

    config.enable_numpy_conversion()

    model.load_state_dict(og_weights)
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
