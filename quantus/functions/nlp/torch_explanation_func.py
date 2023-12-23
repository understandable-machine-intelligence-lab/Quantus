# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Explanation functions for Torch models."""

from __future__ import annotations

from functools import singledispatch, partial, wraps
from importlib import util
from operator import itemgetter
from typing import (
    Callable,
    List,
    Optional,
    Union,
    Protocol,
    NamedTuple,
    Literal,
    Sequence,
    TypeVar,
)

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch import Tensor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances

from quantus.helpers.collection_utils import map_dict
from quantus.helpers.model.torch_hf_model import TorchHuggingFaceTextClassifier
from quantus.helpers.typing_utils import Explanation

if util.find_spec("noisegrad"):
    # Install from https://github.com/aaarrti/NoiseGrad.git
    from noisegrad.noisegrad import (
        NoiseGradConfig,
        NoiseGradPlusPlusConfig,
        NoiseGrad,
        NoiseGradPlusPlus,
    )
else:
    NoiseGradConfig = type(None)
    NoiseGradPlusPlusConfig = type(None)


class BaselineExplainFn(Protocol):
    def __call__(
        self,
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor | list[str],
        y_batch: Tensor,
        **kwargs,
    ) -> Tensor:
        ...


BaselineExplainFn = Union[
    BaselineExplainFn, Literal["GradNorm", "GradXInput", "IntGrad"]
]

T = TypeVar("T")


class LimeConfig(NamedTuple):
    alpha: float = 1.0
    solver: str = "cholesky"
    seed: int = 42
    num_samples: int = 1000
    mask_token: str = "[UNK]"
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = partial(
        pairwise_distances, metric="cosine"
    )
    kernel: Optional[Callable[[float, float], np.ndarray]] = None
    distance_scale: float = 100.0
    kernel_width: float = 25


class TensorConfig(object):
    return_np_arrays: bool = True

    @classmethod
    def enable_numpy_conversion(cls):
        cls.return_np_arrays = True

    @classmethod
    def disable_numpy_conversion(cls):
        cls.return_np_arrays = False


class TensorFunc(Protocol):
    def __call__(
        self,
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor,
        y_batch: Tensor,
        **kwargs,
    ) -> Tensor | np.ndarray:
        ...


class SupportsNumpy(Protocol):
    def __call__(
        self,
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor | np.ndarray,
        y_batch: Tensor | np.ndarray,
        **kwargs,
    ) -> Tensor | np.ndarray:
        ...


class SupportPlainText(Protocol):
    def __call__(
        self,
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor | np.ndarray | List[str],
        y_batch: Tensor | np.ndarray,
        **kwargs,
    ) -> Tensor | np.ndarray | List[Explanation]:
        ...


def tensor_inputs(func: TensorFunc) -> SupportsNumpy:
    @wraps(func)
    def wrapper(
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor | np.ndarray,
        y_batch: Tensor | np.ndarray,
        **kwargs,
    ):
        def map_fn(x):
            if isinstance(x, (np.ndarray, List)):
                return torch.tensor(x, device=model.device)
            else:
                return x

        x_batch = map_fn(x_batch)
        y_batch = map_fn(y_batch)
        kwargs = map_dict(kwargs, map_fn)
        return func(model, x_batch, y_batch, **kwargs)

    return wrapper


def plain_text_inputs(func: SupportsNumpy) -> SupportPlainText:
    @wraps(func)
    def wrapper(
        model: TorchHuggingFaceTextClassifier,
        x_batch: List[str] | Tensor | np.ndarray,
        y_batch: Tensor | np.ndarray,
        **kwargs,
    ):
        if not isinstance(x_batch[0], str):
            return func(model, x_batch, y_batch, **kwargs)

        input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)
        embeddings = model.embedding_lookup(input_ids)
        scores = func(model, embeddings, y_batch, **kwargs)
        return [
            (model.tokenizer.convert_ids_to_tokens(i), j)
            for i, j in zip(input_ids, scores)
        ]

    return wrapper


def pop_device_kwarg(func: T) -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            kwargs.pop("device")
        return func(*args, **kwargs)

    return wrapper


@pop_device_kwarg
@plain_text_inputs
@tensor_inputs
def gradient_norm(
    model: TorchHuggingFaceTextClassifier,
    x_batch: Tensor,
    y_batch: Tensor,
    **kwargs,
) -> Tensor:
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
    logits = model(None, inputs_embeds=x_batch, **kwargs)
    logits_for_class = logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), x_batch)[0]

    scores = torch.linalg.norm(grads, dim=-1)
    if TensorConfig.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


@pop_device_kwarg
@plain_text_inputs
@tensor_inputs
def gradient_x_input(
    model: TorchHuggingFaceTextClassifier,
    x_batch: Tensor,
    y_batch: Tensor,
    **kwargs,
) -> Tensor:
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
    logits = model(None, inputs_embeds=x_batch, **kwargs)
    logits_for_class = logits_for_labels(
        logits, torch.tensor(y_batch, device=model.device)
    )
    grads = torch.autograd.grad(torch.unbind(logits_for_class), x_batch)[0]
    scores = torch.sum(grads * x_batch, dim=-1)
    if TensorConfig.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


@pop_device_kwarg
@plain_text_inputs
@tensor_inputs
def integrated_gradients(
    model: TorchHuggingFaceTextClassifier,
    x_batch: Tensor,
    y_batch: Tensor,
    num_steps: int = 10,
    **kwargs,
) -> Tensor:
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

    def pseudo_interpolate(x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=model.device)
        if not isinstance(x, torch.Tensor):
            return x
        old_shape = x.shape
        return torch.reshape(
            torch.broadcast_to(x, (num_steps, *old_shape)),
            (old_shape[0] * num_steps, old_shape[1]),
        )

    kwargs = map_dict(kwargs, pseudo_interpolate)

    def predict_fn(x):
        return model(None, inputs_embeds=x, **kwargs)

    explainer = IntegratedGradients(predict_fn)
    grads = explainer.attribute(inputs=x_batch, n_steps=num_steps, target=y_batch)

    scores = torch.linalg.norm(grads, dim=-1)

    if TensorConfig.return_np_arrays:
        scores = scores.detach().cpu().numpy()
    return scores


@pop_device_kwarg
@singledispatch
def noise_grad(
    model: TorchHuggingFaceTextClassifier,
    x_batch: List[str] | np.ndarray | Tensor,
    y_batch: Tensor | np.ndarray,
    explain_fn: BaselineExplainFn = "IntGrad",
    ng_config: NoiseGradConfig | None = None,
    **kwargs,
) -> np.ndarray | List[Explanation]:
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
    ng_config:
        config passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    if not util.find_spec("noisegrad"):
        raise ValueError("NoiseGrad requires `noisegrad` package installation")
    return _noise_grad(
        x_batch,
        model,
        y_batch,
        explain_fn=explain_fn,
        ng_config=ng_config,
        **kwargs,
    )


@pop_device_kwarg
@plain_text_inputs
@tensor_inputs
def noise_grad_plus_plus(
    model: TorchHuggingFaceTextClassifier,
    x_batch: Tensor,
    y_batch: Tensor,
    *,
    explain_fn: BaselineExplainFn = "IntGrad",
    config: NoiseGradPlusPlusConfig | None = None,
    **kwargs,
) -> Tensor:
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
    config:
        Config passed to __init__ method of NoiseGrad class.

    Returns
    -------

    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    if not util.find_spec("noisegrad"):
        raise ValueError("NoiseGrad++ requires `noisegrad` package installation")
    og_weights = model.state_dict().copy()
    explain_fn = resolve_noise_grad_baseline_explain_fn(explain_fn)
    if is_builtin_explain_fn(explain_fn):
        TensorConfig.disable_numpy_conversion()

    def adapter(module, inputs, targets):
        model.load_state_dict(module.state_dict())
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return base_scores

    ng_pp = NoiseGradPlusPlus(config)
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

    if is_builtin_explain_fn(explain_fn):
        TensorConfig.enable_numpy_conversion()

    model.load_state_dict(og_weights)
    return scores


# ----------------------- NoiseGrad -------------------------


@pop_device_kwarg
@singledispatch
def _noise_grad(
    x_batch: list,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    *,
    explain_fn: BaselineExplainFn,
    config: NoiseGradConfig | None = None,
    **kwargs,
) -> list[Explanation]:
    explain_fn = resolve_noise_grad_baseline_explain_fn(explain_fn)

    baseline_tokens = explain_fn(model, x_batch, y_batch)  # type: ignore
    baseline_tokens = list(map(itemgetter(0), baseline_tokens))

    og_weights = model.state_dict().copy()
    if is_builtin_explain_fn(explain_fn):
        TensorConfig.disable_numpy_conversion()

    def adapter(module, inputs, targets) -> torch.Tensor:
        model.load_state_dict(module.state_dict())
        a_batch = explain_fn(model, x_batch, targets)
        base_scores = torch.stack([i[1] for i in a_batch])
        return base_scores  # noqa

    ng = NoiseGrad(config)
    scores = (
        ng.enhance_explanation(
            model.get_model(),
            x_batch,
            y_batch,
            explanation_fn=adapter,
        )
        .detach()
        .cpu()
        .numpy()
    )
    if is_builtin_explain_fn(explain_fn):
        TensorConfig.enable_numpy_conversion()
    model.load_state_dict(og_weights)
    return list(zip(baseline_tokens, scores))


@pop_device_kwarg
@_noise_grad.register(Tensor)
@_noise_grad.register(np.ndarray)
def _(
    x_batch: Tensor,
    model: TorchHuggingFaceTextClassifier,
    y_batch: Tensor,
    explain_fn: BaselineExplainFn,
    config: NoiseGradConfig | None = None,
    **kwargs,
) -> np.ndarray:
    og_weights = model.state_dict().copy()
    explain_fn = resolve_noise_grad_baseline_explain_fn(explain_fn)

    if is_builtin_explain_fn(explain_fn):
        TensorConfig.disable_numpy_conversion()

    def adapter(module, inputs, targets):
        model.load_state_dict(module.state_dict())
        base_scores = explain_fn(model, inputs, targets, **kwargs)
        return base_scores

    ng_pp = NoiseGrad(config)
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

    if is_builtin_explain_fn(explain_fn):
        TensorConfig.enable_numpy_conversion()
    model.load_state_dict(og_weights)
    return scores


@pop_device_kwarg
def explain_lime(
    model: TorchHuggingFaceTextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    config: Optional[LimeConfig] = None,
) -> List[Explanation]:
    """
    LIME explains classifiers by returning a feature attribution score
    for each input feature. It works as follows:

    1) Sample perturbation masks. First the number of masked features is sampled
        (uniform, at least 1), and then that number of features are randomly chosen
        to be masked out (without replacement).
    2) Get predictions from the model for those perturbations. Use these as labels.
    3) Fit a linear model to associate the input positions indicated by the binary
        mask with the resulting predicted label.

    The resulting feature importance scores are the linear model coefficients for
    the requested output class or (in case of regression) the output score.

    This is a reimplementation of the original https://github.com/marcotcr/lime
    and is tested for compatibility. This version supports applying LIME to text input.

    Returns
    -------

    """
    if config is None:
        config = LimeConfig()

    a_batch = []
    input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)

    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        ids = input_ids[i]
        tokens = model.tokenizer.convert_ids_to_tokens(ids)
        masks = sample_masks(config.num_samples + 1, len(tokens), seed=config.seed)
        assert (
            masks.shape[0] == config.num_samples + 1
        ), "Expected num_samples + 1 masks."
        all_true_mask = np.ones_like(masks[0], dtype=bool)
        masks[0] = all_true_mask

        perturbations = get_perturbations(tokens, masks, config.mask_token)
        logits = model.predict(perturbations)
        outputs = logits[:, y]
        # fmt: off
        distances = config.distance_fn(all_true_mask.reshape(1, -1), masks).flatten()  # noqa
        # fmt: on
        distances = config.distance_scale * distances
        distances = exponential_kernel(distances, config.kernel_width)

        # Fit a linear model for the requested output class.
        local_surrogate_model = Ridge(
            alpha=config.alpha, solver=config.solver, random_state=config.seed
        ).fit(masks, outputs, sample_weight=distances)

        score = local_surrogate_model.coef_  # noqa
        a_batch.append((tokens, score))
    return a_batch


# -------------- utils ------------------


def resolve_noise_grad_baseline_explain_fn(
    explain_fn: str | BaselineExplainFn,
) -> BaselineExplainFn:
    if isinstance(explain_fn, Callable):
        return explain_fn

    if explain_fn in ("NoiseGrad", "NoiseGrad++"):
        raise ValueError(f"Can't use {explain_fn} as baseline function for NoiseGrad.")
    method_mapping = {
        "GradNorm": gradient_norm,
        "GradXInput": gradient_x_input,
        "IntGrad": integrated_gradients,
    }
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]


def logits_for_labels(logits: Tensor, y_batch: Tensor) -> Tensor:
    return logits[torch.arange(0, logits.shape[0], dtype=torch.int), y_batch]


def sample_masks(num_samples: int, num_features: int, seed: Optional[int] = None):
    rng = np.random.RandomState(seed)
    positions = np.tile(np.arange(num_features), (num_samples, 1))
    permutation_fn = np.vectorize(rng.permutation, signature="(n)->(n)", cache=True)
    permutations = permutation_fn(positions)  # A shuffled range of positions.
    num_disabled_features = rng.randint(1, num_features + 1, (num_samples, 1))
    return permutations >= num_disabled_features


def get_perturbations(
    tokens: Sequence[str], masks: np.ndarray, mask_token: str
) -> List[str]:
    """Returns strings with the masked tokens replaced with `mask_token`."""
    result = []
    for mask in masks:
        parts = [t if mask[i] else mask_token for i, t in enumerate(tokens)]
        result.append(" ".join(parts))
    return result


def exponential_kernel(distance: float, kernel_width: float) -> np.ndarray:
    return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def is_builtin_explain_fn(func: BaselineExplainFn) -> bool:
    return func in (gradient_norm, gradient_x_input, integrated_gradients)
