"""Explanation functions for Torch models."""

from __future__ import annotations

from functools import singledispatch
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import get_input_ids, map_dict, value_or_default

__all__ = ["available_xai_methods", "torch_explain"]

# Just to save some typing effort
_BaselineFn = Callable[[Tensor], Tensor]
_TextOrVector = Union[List[str], Tensor]
_Scores = Union[List[Explanation], Tensor]

# ----------------- "API" --------------------


def available_xai_methods() -> Dict:
    return {
        "GradNorm": explain_gradient_norm,
        "GradXInput": explain_gradient_x_input,
        "IntGrad": explain_integrated_gradients,
        "NoiseGrad++": explain_noise_grad_plus_plus,
        "NoiseGrad": explain_noise_grad,
    }


def torch_explain(
    *args,
    method: str,
    **kwargs,
) -> _Scores:
    method_mapping = available_xai_methods()

    if method not in method_mapping:
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(method_mapping.keys())}"
        )
    explain_fn = method_mapping[method]  # type: ignore
    return explain_fn(*args, **kwargs)


# --------- Multiple dispatch stubs, which allos calling XAI methods on plain-text input and embeddings -----------


def explain_gradient_norm(
    model: TextClassifier, x_batch: _TextOrVector, y_batch: np.ndarray, **kwargs
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
    return _explain_gradient_norm(x_batch, model, y_batch, **kwargs)


def explain_gradient_x_input(
    model: TextClassifier, x_batch: _TextOrVector, y_batch: np.ndarray, **kwargs
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
    return _explain_gradient_x_input(x_batch, model, y_batch, **kwargs)


def explain_integrated_gradients(
    model: TextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> _Scores:
    """
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
    baseline_fn:
        Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
        embedding for [UNK] token could be used.
    batch_interpolated_inputs:
        Indicates if interpolated inputs should be stacked into 1 bigger batch.
        This speeds up the explanation, however can be very memory intensive.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    >>> def unknown_token_baseline_function(x):
        ... return Tensor(np.load(...), dtype=torch.float32).to(device) # noqa

    >>> torch_explain_integrated_gradients(..., ..., ..., baseline_fn=unknown_token_baseline_function) # noqa

    """
    return _explain_integrated_gradients(
        x_batch,
        model,
        y_batch,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
        batch_interpolated_inputs=batch_interpolated_inputs,
        **kwargs,
    )


def explain_noise_grad(
    model: TextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    *,
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
        Otherwise, must have `Callable[[TextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    init_kwargs:
        Kwargs passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _explain_noise_grad(
        x_batch,
        model,
        y_batch,
        explain_fn=explain_fn,
        init_kwargs=init_kwargs,
        **kwargs,
    )


def explain_noise_grad_plus_plus(
    model: TextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
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
        Otherwise, must have `Callable[[TextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    init_kwargs:
        Kwargs passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _explain_noise_grad_plus_plus(
        x_batch,
        model,
        y_batch,
        explain_fn=explain_fn,
        init_kwargs=init_kwargs,
        **kwargs,
    )


# ------------ single dispatch stubs  --------------


@singledispatch
def _explain_gradient_norm(
    x_batch, model: TextClassifier, y_batch: np.ndarray, **kwargs
) -> _Scores:
    pass


@singledispatch
def _explain_gradient_x_input(
    x_batch, model: TextClassifier, y_batch: np.ndarray, **kwargs
) -> _Scores:
    pass


@singledispatch
def _explain_integrated_gradients(
    x_batch,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> _Scores:
    pass


@singledispatch
def _explain_noise_grad(
    x_batch,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> _Scores:
    pass


@singledispatch
def _explain_noise_grad_plus_plus(
    x_batch,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> _Scores:
    pass


# ----------------------- GradNorm -------------------------


@_explain_gradient_norm.register
def _(
    x_batch: list, model: TextClassifier, y_batch: np.ndarray, **kwargs
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _explain_gradient_norm(input_embeds, model, y_batch, **kwargs)
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_explain_gradient_norm.register
def _(x_batch: np.ndarray, model: TextClassifier, *args, **kwargs):
    return _explain_gradient_norm(model.to_tensor(x_batch), model, *args, **kwargs)


@_explain_gradient_norm.register
def _(
    x_batch: torch.Tensor,
    model: TextClassifier,
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """A version of GradientNorm explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""

    device = model.device
    dtype = model.unwrap().dtype

    input_embeddings = torch.tensor(
        x_batch, requires_grad=True, device=device, dtype=dtype
    )

    kwargs = map_dict(kwargs, lambda x: torch.tensor(x, device=device))
    logits = model(input_embeddings, **kwargs)
    logits_for_class = get_logits_for_labels(logits, model.to_tensor(y_batch))
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.linalg.norm(grads, dim=-1).detach().cpu().numpy()


# ----------------------- GradXInput -------------------------


@_explain_gradient_x_input.register
def _(
    x_batch: list, model: TextClassifier, y_batch: np.ndarray, **kwargs
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)
    scores = _explain_gradient_x_input(input_embeds, model, y_batch, **kwargs)

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_explain_gradient_x_input.register
def _(x_batch: np.ndarray, model: TextClassifier, *args, **kwargs):
    return _explain_gradient_x_input(model.to_tensor(x_batch), model, *args, **kwargs)


@_explain_gradient_x_input.register
def _(
    x_batch: Tensor,
    model: TextClassifier,
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """A version of GradientXInput explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""

    device = model.device
    dtype = model.unwrap().dtype
    input_embeddings = torch.tensor(
        x_batch, requires_grad=True, device=device, dtype=dtype
    )
    kwargs = map_dict(kwargs, lambda x: torch.tensor(x, device=device))
    logits = model(input_embeddings, **kwargs)
    logits_for_class = get_logits_for_labels(logits, model.to_tensor(y_batch))
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach().cpu().numpy()


# ----------------------- IntGrad -------------------------


@_explain_integrated_gradients.register
def _(
    x_batch: list,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    input_embeds = model.embedding_lookup(input_ids)

    input_embeds = input_embeds.detach().cpu().numpy()

    scores = _explain_integrated_gradients(
        input_embeds,
        model,
        y_batch,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
        batch_interpolated_inputs=batch_interpolated_inputs,
        **kwargs,
    )

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_explain_integrated_gradients.register
def _(x_batch: np.ndarray, model: TextClassifier, *args, **kwargs):
    return _explain_integrated_gradients(
        model.to_tensor(x_batch), model, *args, **kwargs
    )


@_explain_integrated_gradients.register
def _(
    x_batch: np.ndarray,
    model: TextClassifier,
    y_batch: np.asarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> np.ndarray:
    baseline_fn = value_or_default(baseline_fn, lambda: lambda x: np.zeros_like(x))

    interpolated_embeddings = []

    for i, embeddings_i in enumerate(x_batch):
        interpolated_embeddings.append(
            get_interpolated_inputs(baseline_fn(embeddings_i), embeddings_i, num_steps)
        )

    if batch_interpolated_inputs:
        return _torch_explain_integrated_gradients_batched(
            model, interpolated_embeddings, y_batch, **kwargs
        )
    else:
        return _torch_explain_integrated_gradients_iterative(
            model, interpolated_embeddings, y_batch, **kwargs
        )


def _torch_explain_integrated_gradients_batched(
    model: TextClassifier,
    interpolated_embeddings: List[np.ndarray],
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    device = model.device

    batch_size = len(interpolated_embeddings)
    num_steps = len(interpolated_embeddings[0])

    dtype = model.unwrap().dtype

    interpolated_embeddings = torch.tensor(
        interpolated_embeddings, requires_grad=True, device=device, dtype=dtype
    )
    interpolated_embeddings = torch.reshape(
        interpolated_embeddings, [-1, *interpolated_embeddings.shape[2:]]  # type: ignore
    )

    def pseudo_interpolate(x):
        x = np.broadcast_to(x, (num_steps, *x.shape))
        x = np.reshape(x, (-1, *x.shape[2:]))
        return x

    interpolated_kwargs = map_dict(kwargs, pseudo_interpolate)
    logits = model(interpolated_embeddings, **interpolated_kwargs)
    logits_for_class = get_logits_for_labels(
        logits, model.to_tensor(pseudo_interpolate(y_batch))
    )
    grads = torch.autograd.grad(torch.unbind(logits_for_class), interpolated_embeddings)
    grads = grads[0]
    grads = torch.reshape(grads, [batch_size, num_steps, *grads.shape[1:]])
    scores = torch.trapz(torch.trapz(grads, dim=-1), dim=1)
    return scores.detach().cpu().numpy()


def _torch_explain_integrated_gradients_iterative(
    model: TextClassifier,
    interpolated_embeddings_batch: List[np.ndarray],
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    dtype = model.unwrap().dtype

    device = model.device
    scores = []

    for i, interpolated_embeddings in enumerate(interpolated_embeddings_batch):
        interpolated_embeddings = torch.tensor(
            interpolated_embeddings, requires_grad=True, device=device, dtype=dtype
        )

        interpolated_kwargs = map_dict(
            {k: v[i] for k, v in kwargs.items()},
            lambda x: np.broadcast_to(x, (interpolated_embeddings.shape[0], *x.shape)),
        )

        logits = model(interpolated_embeddings, **interpolated_kwargs)
        logits_for_class = logits[:, y_batch[i]]
        grads = torch.autograd.grad(
            torch.unbind(logits_for_class), interpolated_embeddings
        )
        score = torch.trapz(torch.trapz(grads[0], dim=-1), axis=0)

        scores.append(score.detach().cpu().numpy())

    return np.asarray(scores)


# ----------------------- NoiseGrad -------------------------


@_explain_noise_grad.register
def _(
    x_batch: list,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> List[Explanation]:
    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    input_ids, kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)

    embeddings = embeddings.detach().cpu().numpy()

    scores = _explain_noise_grad(
        embeddings, model, y_batch, explain_fn, init_kwargs, **kwargs
    )
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_explain_noise_grad.register
def _(x_batch: np.ndarray, model: TextClassifier, *args, **kwargs):
    return _explain_gradient_x_input(model.to_tensor(x_batch), model, *args, **kwargs)


@_explain_noise_grad.register
def _(
    x_batch: np.ndarray,
    model: TextClassifier,
    y_batch: np.ndarray,
    explain_fn: Callable,
    init_kwargs: Optional[Dict],
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGrad

    device = model.device
    x_batch = torch.tensor(x_batch, device=device)
    y_batch = torch.tensor(y_batch, device=device)

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    og_weights = model.weights.copy()

    def explanation_fn(
        module: torch.nn.Module, inputs: Tensor, targets: Tensor
    ) -> Tensor:
        model.weights = module.state_dict()
        # fmt: off
        np_scores = explain_fn(model, inputs, targets, **kwargs)
        # fmt: on
        return Tensor(np_scores, device=device)

    ng_pp = NoiseGrad(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.unwrap(),
            x_batch,
            y_batch,
            explanation_fn=explanation_fn,
        )
        .detach()
        .cpu()
        .numpy()
    )

    model.weights = og_weights
    return scores


# ----------------------- NoiseGrad++ -------------------------


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


@_explain_noise_grad_plus_plus.register
def _(
    x_batch: list,
    model: TextClassifier,
    y_batch: np.ndarray,
    *,
    explain_fn: Union[Callable, str] = "IntGrad",
    init_kwargs: Optional[Dict] = None,
    **kwargs,
) -> List[Explanation]:
    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    input_ids, kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)

    embeddings = embeddings.detach().cpu().numpy()

    scores = _explain_noise_grad_plus_plus(
        embeddings, model, y_batch, explain_fn, init_kwargs, **kwargs
    )
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_explain_noise_grad_plus_plus.register
def _(x_batch: np.ndarray, model: TextClassifier, *args, **kwargs):
    return _explain_gradient_x_input(model.to_tensor(x_batch), model, *args, **kwargs)


@_explain_noise_grad_plus_plus.register
def _(
    x_batch: np.ndarray,
    model: TextClassifier,
    y_batch: np.ndarray,
    explain_fn: Callable,
    init_kwargs: Optional[Dict],
    **kwargs,
) -> np.ndarray:
    from noisegrad import NoiseGradPlusPlus

    device = model.device
    x_batch = torch.tensor(x_batch, device=device)
    y_batch = torch.tensor(y_batch, device=device)

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    og_weights = model.weights.copy()

    def explanation_fn(
        module: torch.nn.Module, inputs: Tensor, targets: Tensor
    ) -> Tensor:
        model.weights = module.state_dict()
        # fmt: off
        np_scores = explain_fn(model, inputs, targets, **kwargs)
        # fmt: on
        return Tensor(np_scores, device=device)

    ng_pp = NoiseGradPlusPlus(**init_kwargs)
    scores = (
        ng_pp.enhance_explanation(
            model.unwrap(),
            x_batch,
            y_batch,
            explanation_fn=explanation_fn,
        )
        .detach()
        .cpu()
        .numpy()
    )

    model.weights = og_weights
    return scores


# -------------- utils ------------------


def get_logits_for_labels(logits: Tensor, y_batch: Tensor) -> Tensor:
    return logits[torch.range(0, logits.shape[0] - 1, dtype=torch.int64), y_batch]


def get_interpolated_inputs(
    baseline: np.ndarray, target: np.ndarray, num_steps: int
) -> np.ndarray:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    if num_steps <= 0:
        return np.array([])
    if num_steps == 1:
        return np.array([baseline, target])

    delta = target - baseline
    scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[
        :, np.newaxis, np.newaxis
    ]
    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs
