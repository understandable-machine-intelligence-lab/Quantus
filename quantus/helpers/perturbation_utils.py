from __future__ import annotations

import sys
from typing import List, TYPE_CHECKING, Callable, Mapping, Optional
import numpy as np
import functools

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


if TYPE_CHECKING:
    from quantus.helpers.model.model_interface import ModelInterface

    class PerturbFunc(Protocol):
        def __call__(
            self,
            arr: np.ndarray,
            **kwargs,
        ) -> np.ndarray: ...


def make_perturb_func(
    perturb_func: PerturbFunc, perturb_func_kwargs: Mapping[str, ...] | None, **kwargs
) -> PerturbFunc | functools.partial:
    """A utility function to save few lines of code during perturbation metric initialization."""
    if perturb_func_kwargs is not None:
        func_kwargs = kwargs.copy()
        func_kwargs.update(perturb_func_kwargs)
    else:
        func_kwargs = kwargs

    return functools.partial(perturb_func, **func_kwargs)


def make_changed_prediction_indices_func(
    return_nan_when_prediction_changes: bool,
) -> Callable[[ModelInterface, np.ndarray, np.ndarray], List[int]]:
    """A utility function to improve static analysis."""
    return functools.partial(
        changed_prediction_indices,
        return_nan_when_prediction_changes=return_nan_when_prediction_changes,
    )


def changed_prediction_indices(
    model: ModelInterface,
    x_batch: np.ndarray,
    x_perturbed: np.ndarray,
    return_nan_when_prediction_changes: bool,
) -> List[int]:
    """
    Find indices in batch, for which predicted label has changed after applying perturbation.
    If metric `return_nan_when_prediction_changes` is False, will return empty list.

    Parameters
    ----------
    return_nan_when_prediction_changes:
        Instance attribute of perturbation metrics.
    model:
    x_batch:
        Batch of original inputs provided by user.
    x_perturbed:
        Batch of inputs after applying perturbation.

    Returns
    -------

    changed_idx:
        List of indices in batch, for which predicted label has changed afer.

    """

    if not return_nan_when_prediction_changes:
        return []

    labels_before = model.predict(x_batch).argmax(axis=-1)
    labels_after = model.predict(x_perturbed).argmax(axis=-1)
    changed_idx = np.reshape(np.argwhere(labels_before != labels_after), -1)
    return changed_idx.tolist()
