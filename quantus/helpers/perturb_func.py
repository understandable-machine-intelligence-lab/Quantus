"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import copy
import random
import warnings
from typing import Any, Sequence, Tuple, Union

import cv2
import numpy as np
import scipy

from .utils import get_baseline_value
from .utils import conv2D_numpy


def gaussian_noise(
    arr: np.array, perturb_mean: float = 0.0, perturb_std: float = 0.01, **kwargs
) -> np.array:
    """Add gaussian noise to the input."""
    noise = np.random.normal(loc=perturb_mean, scale=perturb_std, size=arr.shape)
    return arr + noise

def baseline_replacement_by_indices(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indices_axis: Sequence[int],
    replacement_value: Union[float, int, str, np.array],
    **kwargs,
) -> np.array:
    """
    Replace indices in an array by a given baseline.
    indices: array-like, with a subset shape of arr
    indices_axis: dimensions of arr that are indexed. These need to be consecutive,
                  and either include the first or last dimension of array.
    """
    indices = np.array(indices)
    indices_axis = sorted(indices_axis)

    # TODO @Leander: include this into asserts?
    assert list(indices_axis) == list(range(indices_axis[0], indices_axis[-1]+1))
    assert 0 in indices_axis or arr.ndim-1 in indices_axis

    if len(indices_axis) != indices.ndim:
        if indices.ndim == 1:
            indices = np.unravel_index(indices, tuple([arr.shape[i] for i in indices_axis]))
        else:
            raise ValueError("indices dimension doesn't match indices_axis")

    baseline_shape = []
    # Indices First
    if 0 in indices_axis:
        for i in range(indices_axis[-1]+1, arr.ndim):
            indices = indices, slice(None)
            baseline_shape.append(arr.shape[i])
    # Indices Last
    else:
        for i in range(0, indices_axis[0]):
            indices = slice(None), indices
            baseline_shape.append(arr.shape[i])

    arr_perturbed = copy.copy(arr)

    # Get Baseline
    baseline_value = get_baseline_value(
        value=replacement_value,
        arr=arr,
        return_shape=tuple(baseline_shape),
        **kwargs
    )

    arr_perturbed[indices] = baseline_value
    return arr_perturbed

def baseline_replacement_by_shift(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indices_axis: Sequence[int],
    shift_value: Union[float, int, str, np.array],
    ** kwargs,
) -> np.array:
    """
    Shift values at indices in an image.
    indices: array-like, with a subset shape of arr
    indices_axis: axes of arr that are indexed. These need to be consecutive,
                  and either include the first or last dimension of array.
    """
    indices = np.array(indices)
    indices_axis = sorted(indices_axis)

    # TODO @Leander: include this into asserts?
    assert list(indices_axis) == list(range(indices_axis[0], indices_axis[-1]+1))
    assert 0 in indices_axis or arr.ndim-1 in indices_axis

    if len(indices_axis) != indices.ndim:
        if indices.ndim == 1:
            indices = np.unravel_index(indices, tuple([arr.shape[i] for i in indices_axis]))
        else:
            raise ValueError("indices dimension doesn't match indices_axis")

    baseline_shape = []
    # Indices First
    if 0 in indices_axis:
        for i in range(indices_axis[-1]+1, arr.ndim):
            indices = indices, slice(None)
            baseline_shape.append(arr.shape[i])
    # Indices Last
    else:
        for i in range(0, indices_axis[0]):
            indices = slice(None), indices
            baseline_shape.append(arr.shape[i])

    arr_perturbed = copy.copy(arr)

    # Get Baseline
    baseline_value = get_baseline_value(
        value=shift_value,
        arr=arr,
        return_shape=tuple(baseline_shape),
        **kwargs
    )

    # Shift
    arr_shifted = copy.copy(arr_perturbed)
    expand_axis = (dim for dim in range(baseline_value.ndim, arr_shifted.ndim))
    arr_shifted = np.add(
        arr_shifted,
        np.full(
            shape=arr_shifted.shape,
            fill_value=np.expand_dims(baseline_value, axis=tuple(indices_axis)),
            dtype=float
        ),
    )

    arr_perturbed[indices] = arr_shifted[indices]
    return arr_perturbed

def baseline_replacement_by_blur(
    arr: np.array, indices: Union[int, Sequence[int], Tuple[np.array]], blur_kernel_size: int = 15, **kwargs
) -> np.array:
    """
    Replace a single patch in an array by a blurred version.
    Blur is performed via a 2D convolution.
    blur_kernel_size controls the kernel-size of that convolution (Default is 15).
    Assumes unbatched channel first format.
    """

    # TODO @Leander: change this for arbitrary input shapes
    # TODO @Leander: double check axes
    nr_channels = arr.shape[0]
    # Create blurred array.
    blur_kernel_size = (1, *([blur_kernel_size] * (arr.ndim - 1)))
    kernel = np.ones(blur_kernel_size, dtype=arr.dtype)
    kernel *= 1.0 / np.prod(blur_kernel_size)
    kernel = np.tile(kernel, (nr_channels, 1, *([1] * (arr.ndim - 1))))

    if arr.ndim == 3:
        # TODO @Leander: Check return shape for different kernel sizes
        arr_avg = conv2D_numpy(
            x=arr,
            kernel=kernel,
            stride=1,
            padding=0,
            groups=nr_channels,
            pad_output=True,
        )
    elif arr.ndim == 2:
        raise NotImplementedError("1d support not implemented yet")
    else:
        raise ValueError("Blur supports only 2d inputs")

    # Perturb array.
    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = arr_avg[indices]
    return arr_perturbed


def uniform_sampling(arr: np.array, perturb_radius: float = 0.02, **kwargs) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    noise = np.random.uniform(low=-perturb_radius, high=perturb_radius, size=arr.shape)
    return arr + noise

def rotation(arr: np.array, perturb_angle: float = 10, **kwargs) -> np.array:
    """
    Rotate array by some given angle.
    Assumes channel first layout.
    """
    if arr.ndim != 3:
        raise ValueError("Check that 'perturb_func' receives a 3D array.")

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
    arr: np.array, perturb_value: Any, perturb_dx: int = 10, **kwargs
) -> np.array:
    """
    Translate array by some given value in the x-direction.
    Assumes channel first layout.
    """
    # TODO @Leander: arbitrary shapes?
    if arr.ndim != 3:
        raise ValueError("Check that 'perturb_func' receives a 3D array.")

    matrix = np.float32([[1, 0, perturb_dx], [0, 1, 0]])
    # TODO @Leander: return shape for baseline value?
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, -1),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_value,
            arr=arr,
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
    Assumes channel first layout.
    """
    # TODO @Leander: arbitrary shapes?
    if arr.ndim != 3:
        raise ValueError("Check that 'perturb_func' receives a 3D array.")

    matrix = np.float32([[1, 0, 0], [0, 1, perturb_dx]])
    # TODO @Leander: return shape for baseline value?
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            value=perturb_baseline,
            arr=arr,
            **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
