"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import copy
import random
from typing import Any, Sequence

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


def baseline_replacement_by_indices(arr: np.array, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    assert arr.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    assert (
        "indices" in kwargs
    ), "Specify 'indices' to enable perturbation function to run."
    if "fixed_values" in kwargs:
        choice = kwargs["fixed_values"]
    elif "perturb_baseline" in kwargs:
        choice = kwargs["perturb_baseline"]
    elif "input_shift" in kwargs:
        choice = kwargs["input_shift"]

    arr_perturbed = copy.copy(arr)
    baseline_value = get_baseline_value(choice=choice, arr=arr, **kwargs)

    if "input_shift" in kwargs:
        arr_shifted = copy.copy(arr)
        arr_shifted = np.multiply(
            arr_shifted,
            np.full(shape=arr.shape, fill_value=baseline_value, dtype=float),
        )
        arr_perturbed[kwargs["indices"]] = arr_shifted[kwargs["indices"]]
    else:
        arr_perturbed[kwargs["indices"]] = baseline_value

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


def rotation(img: np.array, **kwargs) -> np.array:
    """Rotate image by some given angle."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = cv2.getRotationMatrix2D(
        center=(
            kwargs.get("img_size", 224) / 2,
            kwargs.get("img_size", 224) / 2,
        ),
        angle=kwargs.get("perturb_angle", 10),
        scale=1,
    )
    img_perturbed = np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
        ),
        2,
        0,
    )
    return img_perturbed


def translation_x_direction(img: np.array, **kwargs) -> np.array:
    """Translate image by some given value in the x-direction."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = np.float32([[1, 0, kwargs.get("perturb_dx", 10)], [0, 1, 0]])
    img_perturbed = np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, -1),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            borderValue=get_baseline_value(
                choice=kwargs["perturb_baseline"], arr=img, **kwargs
            ),
        ),
        -1,
        0,
    )
    return img_perturbed


def translation_y_direction(img: np.array, **kwargs) -> np.array:
    """Translate image by some given value in the x-direction."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = np.float32([[1, 0, 0], [0, 1, kwargs.get("perturb_dy", 10)]])
    img_perturbed = np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            borderValue=get_baseline_value(
                choice=kwargs["perturb_baseline"], arr=img, **kwargs
            ),
        ),
        2,
        0,
    )
    return img_perturbed


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
