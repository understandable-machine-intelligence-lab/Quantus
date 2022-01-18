"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2
import copy
import random
from .utils import *


def gaussian_noise(img: np.array, **kwargs) -> np.array:
    """Add gaussian noise to the input."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    img_perturbed = img + np.random.normal(
        kwargs.get("perturb_mean", 0.0), kwargs.get("perturb_std", 0.01)
    )
    return img_perturbed


def baseline_replacement_by_indices(img: np.array, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    assert (
        "indices" in kwargs
    ), "Specify 'indices' to enable perturbation function to run."
    if "fixed_values" in kwargs:
        choice = kwargs["fixed_values"]
    elif "perturb_baseline" in kwargs:
        choice = kwargs["perturb_baseline"]
    elif "input_shift" in kwargs:
        choice = kwargs["input_shift"]

    img_perturbed = copy.copy(img)
    baseline_value = get_baseline_value(choice=choice, img=img, **kwargs)

    if "input_shift" in kwargs:
        img_shifted = copy.copy(img)
        img_shifted = np.multiply(
            img_shifted,
            np.full(shape=img.shape, fill_value=baseline_value, dtype=float),
        )
        img_perturbed[kwargs["indices"]] = img_shifted[kwargs["indices"]]
    else:
        img_perturbed[kwargs["indices"]] = baseline_value

    return img_perturbed


def baseline_replacement_by_patch(img: np.array, **kwargs) -> np.array:
    """Replace a single patch in an image by given baseline."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "patch_size" in kwargs, "Specify 'patch_size' (int) to perturb the image."
    assert "nr_channels" in kwargs, "Specify 'nr_channels' (int) to perturb the image."
    assert "top_left_y" in kwargs, "Specify 'top_left_y' (int) to perturb the image."
    assert "top_left_x" in kwargs, "Specify 'top_left_x' (int) to perturb the image."

    # Preset patch for 'mean' and 'neighbourhood' choices.
    kwargs["patch"] = img[
        :,
        kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"],
        kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"],
    ]

    img_perturbed = copy.copy(img)
    img_perturbed[
        :,
        kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"],
        kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"],
    ] = get_baseline_value(choice=kwargs["perturb_baseline"], img=img, **kwargs)

    return img_perturbed


def uniform_sampling(img: np.array, **kwargs) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    img_perturbed = img + np.random.uniform(
        -kwargs.get("perturb_radius", 0.02),
        kwargs.get("perturb_radius", 0.02),
        size=img.shape,
    )
    return img_perturbed


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


def no_perturbation(img: np.array, **kwargs) -> np.array:
    """Apply no perturbation to input."""
    img_perturbed = img
    return img_perturbed
