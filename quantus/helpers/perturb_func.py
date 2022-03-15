"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import copy
import random
from typing import Any, Sequence, Union

import cv2
import numpy as np
import scipy

from .utils import get_baseline_value
from .utils import conv2D_numpy


def gaussian_noise(arr: np.array, perturb_mean: float = 0.0,
                   perturb_std: float = 0.01) -> np.array:
    """Add gaussian noise to the input."""
    noise = np.random.normal(loc=perturb_mean, scale=perturb_std, size=arr.shape)
    return arr + noise


def baseline_replacement_by_indices(arr: np.array, indices: Union[int, Sequence[int]],
                                    nr_channels: int, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    if arr.ndim != 1:
        raise ValueError("Check that 'perturb_func' receives a 1D array.")

    if "fixed_values" in kwargs:
        choice = kwargs["fixed_values"]
    elif "perturb_baseline" in kwargs:
        choice = kwargs["perturb_baseline"]
    elif "input_shift" in kwargs:
        choice = kwargs["input_shift"]

    arr_perturbed = copy.copy(arr)
    baseline_value = get_baseline_value(choice=choice, arr=arr, **kwargs)

    # Make sure that image is perturbed on all channels.
    if isinstance(indices, int):
        indices = np.expand_dims(indices, axis=0)
    indices = np.concatenate(
        [
            np.add(
                indices,
                int(c * len(arr) / nr_channels),
                dtype=int,
            )
            for c in range(nr_channels)
        ]
    )

    if "input_shift" in kwargs:
        arr_shifted = copy.copy(arr)
        arr_shifted = np.multiply(
            arr_shifted,
            np.full(shape=arr.shape, fill_value=baseline_value, dtype=float),
        )
        arr_perturbed[indices] = arr_shifted[indices]
    else:
        arr_perturbed[indices] = baseline_value

    return arr_perturbed


def baseline_replacement_by_patch(arr: np.array, patch_slice: Sequence,
                                  perturb_baseline: Any) -> np.array:
    """Replace a single patch in an image by given baseline."""
    if len(patch_slice) != arr.ndim:
        raise ValueError(
            "patch_slice dimensions don't match arr dimensions."
            f" ({len(patch_slice)} != {arr.ndim})"
        )

    # Preset patch for 'neighbourhood_*' choices.
    patch = arr[patch_slice]
    arr_perturbed = copy.copy(arr)
    baseline = get_baseline_value(choice=perturb_baseline, arr=arr, patch=patch)
    arr_perturbed[patch_slice] = baseline
    return arr_perturbed


def baseline_replacement_by_blur(arr: np.array, patch_slice: Sequence,
                                 blur_kernel_size: int = 15) -> np.array:
    """
    Replace a single patch in an image by a blurred version.
    Blur is performed via a 2D convolution.
    blur_kernel_size controls the kernel-size of that convolution (Default is 15).
    Assumes unbatched channel first format.
    """
    nr_channels = arr.shape[0]
    # Create blurred array.
    blur_kernel_size = (1, *([blur_kernel_size] * (arr.ndim - 1)))
    kernel = np.ones(blur_kernel_size, dtype=arr.dtype)
    kernel *= 1.0 / np.prod(blur_kernel_size)
    kernel = np.tile(kernel, (nr_channels, 1, *([1] * (arr.ndim - 1))))

    if arr.ndim == 2:
        raise NotImplementedError()
    elif arr.ndim == 3:
        arr_avg = conv2D_numpy(
            x=arr,
            kernel=kernel,
            stride=1,
            padding=0,
            groups=nr_channels,
            pad_output=True,
        )

    # Perturb array.
    arr_perturbed = copy.copy(arr)
    arr_perturbed[patch_slice] = arr_avg[patch_slice]
    return arr_perturbed


def uniform_sampling(arr: np.array, perturb_radius: float = 0.02) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    noise = np.random.uniform(low=-perturb_radius, high=perturb_radius, size=arr.shape)
    return arr + noise


def rotation(arr: np.array, perturb_angle: float = 10) -> np.array:
    """
    Rotate image by some given angle.
    Assumes channel first layout.
    """
    assert arr.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    matrix = cv2.getRotationMatrix2D(
        center=(arr.shape[1]/2, arr.shape[2]/2),
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


def translation_x_direction(arr: np.array, perturb_baseline: Any,
                            perturb_dx: int = 10, **kwargs) -> np.array:
    """
    Translate image by some given value in the x-direction.
    Assumes channel first layout.
    """
    assert arr.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    matrix = np.float32([[1, 0, perturb_dx], [0, 1, 0]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, -1),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            choice=perturb_baseline, arr=arr, **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, -1, 0)
    return arr_perturbed


def translation_y_direction(arr: np.array, perturb_baseline: Any,
                            perturb_dx: int = 10, **kwargs) -> np.array:
    """
    Translate image by some given value in the x-direction.
    Assumes channel first layout.
    """
    assert arr.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    matrix = np.float32([[1, 0, 0], [0, 1, perturb_dx]])
    arr_perturbed = cv2.warpAffine(
        np.moveaxis(arr, 0, 2),
        matrix,
        (arr.shape[1], arr.shape[2]),
        borderValue=get_baseline_value(
            choice=perturb_baseline, arr=arr, **kwargs,
        ),
    )
    arr_perturbed = np.moveaxis(arr_perturbed, 2, 0)
    return arr_perturbed


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
