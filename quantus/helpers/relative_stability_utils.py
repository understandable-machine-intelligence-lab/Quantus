import numpy as np
from typing import Callable, Optional, Tuple
import warnings
from tqdm.auto import tqdm

from .model_interface import ModelInterface
from .perturb_func import random_noise

"""This module contains common logic shared between Relative Input, Output, Representation Stability metrics"""


def compute_perturbed_inputs_with_same_labels(
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        display_progressbar: bool,
        num_perturbations: int,
        **kwargs
) -> np.ndarray:
    """Computes perturbations which result in the same labels and stack them in new leading axis"""

    device = kwargs.get('device', 'cpu')

    if 'perturb_func' not in kwargs:
        warnings.warn('No "perturb_func" provided, using random noise as default')
        perturb_func = random_noise
    else:
        perturb_func = kwargs.get('perturb_func')

    xs_batch = []
    it = range(num_perturbations)

    if display_progressbar:
        it = tqdm(it, desc="Collecting perturbations")

    for _ in it:
        xs = perturb_func(x_batch, **kwargs)
        logits = model.predict(xs, device=device) # noqa
        labels = np.argmax(logits, axis=1)

        same_label_indexes = np.argwhere(labels == y_batch)
        xs = xs[same_label_indexes].reshape(-1, *xs.shape[1:])
        xs_batch.append(xs)

    # pull all new images into 0 axes
    xs_batch = np.vstack(xs_batch)
    # drop images, which cause dims not to be divisible
    xs_batch = xs_batch[: xs_batch.shape[0] // x_batch.shape[0] * x_batch.shape[0]]
    # make xs_batch have the same shape as x_batch, with new batching axis at 0
    xs_batch = xs_batch.reshape(-1, *x_batch.shape)
    return xs_batch


def assert_correct_kwargs_provided(a_batch, **kwargs):
    if a_batch is not None and "as_batch" in kwargs and "xs_batch" not in kwargs:
        raise ValueError("When providing pre-computed explanations, must also provide x' (xs_batch)")

    if "explain_func" in kwargs and (a_batch is not None or "as_batch" in kwargs):
        raise ValueError("Must provide either explain_func or (a_batch and as_batch)")

    if "explain_func" not in kwargs and (a_batch is None or "as_batch" not in kwargs):
        raise ValueError("Must provide either explain_func or (a_batch and as_batch)")


def compute_explanations(
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        xs_batch: np.ndarray,
        display_progressbar: bool,
        normalize: bool,
        normalize_func: Optional[Callable],
        absolute: bool,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes explanations for the x_batch and xs_batch"""

    # We did assert previously that the correct kwargs were provided,
    # which means if we're here `explain_func` was provided in kwargs
    explain_func: Callable = kwargs.get("explain_func")
    a_batch = explain_func(model=model.get_model(), inputs=x_batch, targets=y_batch, **kwargs)

    it = xs_batch
    if display_progressbar:
        it = tqdm(it, desc=f"Collecting explanations")

    as_batch = [
        explain_func(
            model=model.get_model(), inputs=i, targets=y_batch, **kwargs
        )
        for i in it
    ]

    if normalize:
        a_batch = normalize_func(a_batch)
        as_batch = [normalize_func(i) for i in as_batch]

    if absolute:
        a_batch = np.abs(a_batch)
        as_batch = np.abs(as_batch)

    return a_batch, np.asarray(as_batch)
