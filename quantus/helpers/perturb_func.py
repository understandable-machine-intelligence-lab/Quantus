"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import copy
import random
from typing import Any, Sequence

import cv2
import numpy as np
import scipy

from .utils import conv2D_numpy
from .utils import get_baseline_value


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


def baseline_replacement_by_blur(img: np.array, **kwargs) -> np.array:
    """
    Replace a single patch in an image by a blurred version.
    Blur is performed via a 2D convolution.
    kwarg "blur_patch_size" controls the kernel-size of that convolution (Default is 15).
    """
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "patch_size" in kwargs, "Specify 'patch_size' (int) to perturb the image."
    assert "nr_channels" in kwargs, "Specify 'nr_channels' (int) to perturb the image."
    assert "img_size" in kwargs, "Specify 'img_size' (int) to perturb the image."
    assert "top_left_y" in kwargs, "Specify 'top_left_y' (int) to perturb the image."
    assert "top_left_x" in kwargs, "Specify 'top_left_x' (int) to perturb the image."

    # Get kwargs
    # The patch-size for the blur generation (NOT the patch-size for perturbation)
    blur_patch_size = kwargs.get("blur_patch_size", 15)
    nr_channels = kwargs.get("nr_channels", 3)
    img_size = kwargs.get("img_size", 224)

    # Reshape image since rest of function assumes channels_first
    img = img.reshape(nr_channels, img_size, img_size)

    # Get blurred image
    weightavg = (
        1.0
        / float(blur_patch_size * blur_patch_size)
        * np.ones((1, 1, blur_patch_size, blur_patch_size), dtype=img.dtype)
    )
    print(img.shape, np.tile(weightavg, (nr_channels, 1, 1, 1)).shape)
    avgimg = conv2D_numpy(
        img,
        np.tile(weightavg, (nr_channels, 1, 1, 1)),
        stride=1,
        padding=0,
        groups=nr_channels,
    )
    padwidth = (blur_patch_size - 1) // 2
    avgimg = np.pad(
        avgimg, ((0, 0), (padwidth, padwidth), (padwidth, padwidth)), mode="edge"
    )

    # Perturb image
    img_perturbed = copy.copy(img)
    img_perturbed[
        :,
        kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"],
        kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"],
    ] = avgimg[
        :,
        kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"],
        kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"],
    ]

    return img_perturbed


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
                choice=kwargs["perturb_baseline"], img=img, **kwargs
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
                choice=kwargs["perturb_baseline"], img=img, **kwargs
            ),
        ),
        2,
        0,
    )
    return img_perturbed


def no_perturbation(arr: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    return arr
