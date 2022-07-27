"""This module provides some basic functionality to normalise and denormalise images."""
from typing import Callable, Dict, Optional, Union, Sequence
import numpy as np


def normalise_by_max(
    a: np.ndarray, normalized_axes: Union[None, Sequence[int]] = None, **kwargs
) -> np.ndarray:
    """Normalise attributions by the maximum absolute value of the explanation."""

    # No normalization if a is only zeros.
    if np.all(a == 0.0):
        return a

    # Default normalized_axes.
    if normalized_axes is None:
        normalized_axes = list(range(np.ndim(a)))

    # Cast Sequence to tuple so numpy accepts it.
    normalized_axes = tuple(normalized_axes)

    a /= np.max(np.abs(a), axis=normalized_axes)
    return a


def normalise_by_negative(
    a: np.ndarray, normalized_axes: Union[None, Sequence[int]] = None, **kwargs
) -> np.ndarray:
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""

    # No normalization if a is only zeros.
    if np.all(a == 0.0):
        return a

    # Default normalized_axes.
    if normalized_axes is None:
        normalized_axes = list(range(np.ndim(a)))

    # Cast Sequence to tuple so numpy accepts it.
    normalized_axes = tuple(normalized_axes)

    # Build return array from three cases.
    return_array = np.zeros(a.shape, dtype=a.dtype)

    # Case a.min() >= 0.0.
    return_array = np.where(
        a.min(axis=normalized_axes, keepdims=True) >= 0.0,
        a / a.max(axis=normalized_axes, keepdims=True),
        return_array,
    )

    # Case a.max() <= 0.0.
    return_array = np.where(
        a.max(axis=normalized_axes, keepdims=True) <= 0.0,
        -a / a.min(axis=normalized_axes, keepdims=True),
        return_array,
    )

    # Else.
    return_array = np.where(
        (a.min(axis=normalized_axes, keepdims=True) < 0.0)
        and (a.max(axis=normalized_axes, keepdims=True) > 0.0),
        (a > 0.0) * a / a.max(axis=normalized_axes, keepdims=True)
        - (a < 0.0) * a / a.min(axis=normalized_axes, keepdims=True),
        return_array,
    )

    return return_array


def denormalise(
    img: np.ndarray,
    mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
    std: np.ndarray = np.array([0.229, 0.224, 0.225]),
    **kwargs,
) -> np.ndarray:
    """De-normalise a torch image (using conventional ImageNet values)."""
    return (np.array(img) * std.reshape(-1, 1, 1)) + mean.reshape(-1, 1, 1)
