"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import copy
import random
import warnings
from typing import Any, Sequence, Tuple, Union

import cv2
import numpy as np
import scipy

from .utils import (
    get_baseline_value,
    blur_at_indices,
    expand_indices,
    get_leftover_shape,
)


def baseline_replacement_by_indices(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_baseline: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """
    Replace indices in an array by a given baseline.
    indices: array-like, with a subset shape of arr
    indexed_axes: dimensions of arr that are indexed. These need to be consecutive,
                  and either include the first or last dimension of array.
    """
    indices = expand_indices(arr, indices, indexed_axes)
    baseline_shape = get_leftover_shape(arr, indexed_axes)

    arr_perturbed = copy.copy(arr)

    # Get Baseline
    baseline_value = get_baseline_value(
        value=perturb_baseline, arr=arr, return_shape=tuple(baseline_shape), **kwargs
    )

    # Perturb
    arr_perturbed[indices] = np.expand_dims(baseline_value, axis=tuple(indexed_axes))

    return arr_perturbed


def baseline_replacement_by_shift(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    input_shift: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """
    Shift values at indices in an image.
    indices: array-like, with a subset shape of arr
    indexed_axes: axes of arr that are indexed. These need to be consecutive,
                  and either include the first or last dimension of array.
    """
    indices = expand_indices(arr, indices, indexed_axes)
    baseline_shape = get_leftover_shape(arr, indexed_axes)

    arr_perturbed = copy.copy(arr)

    # Get Baseline
    baseline_value = get_baseline_value(
        value=input_shift, arr=arr, return_shape=tuple(baseline_shape), **kwargs
    )

    # Shift
    arr_shifted = copy.copy(arr_perturbed)
    arr_shifted = np.add(
        arr_shifted,
        np.full(
            shape=arr_shifted.shape,
            fill_value=np.expand_dims(baseline_value, axis=tuple(indexed_axes)),
            dtype=float,
        ),
    )

    arr_perturbed[indices] = arr_shifted[indices]
    return arr_perturbed


def baseline_replacement_by_blur(
    arr: np.array,
    indices: Tuple[np.array],
    indexed_axes: Sequence[int],
    blur_kernel_size: Union[int, Sequence[int]] = 15,
    **kwargs,
) -> np.array:
    """
    Replace array at indices by a blurred version.
    Blur is performed via convolution.
    blur_kernel_size controls the kernel-size of that convolution (Default is 15).
    """

    indices = expand_indices(arr, indices, indexed_axes)

    # Expand blur_kernel_size
    if isinstance(blur_kernel_size, int):
        blur_kernel_size = [blur_kernel_size for _ in indexed_axes]

    assert len(blur_kernel_size) == len(indexed_axes)

    # Create kernel and expand dimensions to arr.ndim
    kernel = np.ones(blur_kernel_size, dtype=arr.dtype)
    kernel *= 1.0 / np.prod(blur_kernel_size)

    # Compute blur array. It is only blurred at indices, otherwise it is equal to arr.
    # We only blur at indices since otherwise n-d convolution can be quite computationally expensive
    arr_perturbed = blur_at_indices(arr, kernel, indices, indexed_axes)

    return arr_perturbed


def gaussian_noise(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_mean: float = 0.0,
    perturb_std: float = 0.01,
    **kwargs,
) -> np.array:
    """Add gaussian noise to the input."""

    indices = expand_indices(arr, indices, indexed_axes)
    noise = np.random.normal(loc=perturb_mean, scale=perturb_std, size=arr.shape)

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed + noise)[indices]

    return arr_perturbed


def uniform_noise(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_radius: float = 0.02,
    **kwargs,
) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""

    indices = expand_indices(arr, indices, indexed_axes)
    noise = np.random.uniform(low=-perturb_radius, high=perturb_radius, size=arr.shape)

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed + noise)[indices]

    return arr_perturbed


def rotation(arr: np.array, perturb_angle: float = 10, **kwargs) -> np.array:
    """
    Rotate array by some given angle.
    Assumes image type data and channel first layout.
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'rotation' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = cv2.getRotationMatrix2D(
        center=(arr.shape[1] / 2, arr.shape[2] / 2),
        angle=perturb_angle,
        scale=1,
    )
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def translation_x_direction(
    arr: np.array, perturb_baseline: Any, perturb_dx: int = 10, **kwargs
) -> np.array:
    """
    Translate array by some given value in the x-direction.
    Assumes image type data and channel first layout.
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'translation_x_direction' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = np.float32([[1, 0, perturb_dx], [0, 1, 0]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, -1),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_baseline,
            arr=arr,
            return_shape=(arr.shape[0]),
            **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, -1, 0)
    return arr_perturbed


def translation_y_direction(
    arr: np.array, perturb_baseline: Any, perturb_dx: int = 10, **kwargs
) -> np.array:
    """
    Translate array by some given value in the x-direction.
    Assumes image type data and channel first layout.
    """
    if arr.ndim != 3:
        raise ValueError(
            "perturb func 'translation_y_direction' requires image-type data."
            "Check that this perturb_func receives a 3D array."
        )

    matrix = np.float32([[1, 0, 0], [0, 1, perturb_dx]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_baseline,
            arr=arr,
            return_shape=(arr.shape[0]),
            **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
