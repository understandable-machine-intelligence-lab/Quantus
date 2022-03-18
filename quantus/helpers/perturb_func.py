"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2
import copy
import random
from .utils import *


def gaussian_noise(img: np.array, **kwargs) -> np.array:
    """Add gaussian noise to the input."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    img_perturbed = img + np.random.normal(
        kwargs.get("perturb_mean", 0.0),
        kwargs.get("perturb_std", 0.01),
        size=img.shape,
    )
    return img_perturbed


def baseline_replacement_by_indices(img: np.array, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "nr_channels" in kwargs, "Specify 'nr_channels' (int) to perturb the image."
    assert (
        "indices" in kwargs
    ), "Specify 'indices' to enable perturbation function to run."
    choice = kwargs["perturb_baseline"]

    img_perturbed = copy.copy(img)
    baseline_value = get_baseline_value(choice=choice, img=img, **kwargs)

    # Make sure that image is perturbed on all channels.
    kwargs["indices"] = (slice(0, kwargs.get("nr_channels")),) + kwargs["indices"]

    if len(img_perturbed[kwargs["indices"]].shape) > 1:
        baseline_value = baseline_value[:, None]

    img_perturbed[kwargs["indices"]] = baseline_value

    return img_perturbed

def baseline_replacement_by_shift(img: np.array, **kwargs) -> np.array:
    """Shift values at indices in an image"""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "nr_channels" in kwargs, "Specify 'nr_channels' (int) to perturb the image."
    assert (
        "indices" in kwargs
    ), "Specify 'indices' to enable perturbation function to run."
    assert (
            "input_shift" in kwargs
    ), "Specify 'input_shift' to enable perturbation function to run."
    choice = kwargs["input_shift"]

    img_perturbed = copy.copy(img)
    baseline_value = get_baseline_value(choice=choice, img=img, **kwargs)

    # Make sure that image is perturbed on all channels.
    kwargs["indices"] = (slice(0, kwargs.get("nr_channels")),) + kwargs["indices"]

    # TODO: this is not a shift, right? Due to the multiplication...
    img_shifted = copy.copy(img)
    img_shifted = np.multiply(
        img_shifted,
        np.full(shape=img.shape, fill_value=baseline_value[:, None, None], dtype=float),
    )
    img_perturbed[kwargs["indices"]] = img_shifted[kwargs["indices"]]

    return img_perturbed


def baseline_replacement_by_blur(img: np.array, **kwargs) -> np.array:
    """
    Replace a single patch in an image by a blurred version.
    Blur is performed via a 2D convolution.
    kwarg "blur_patch_size" controls the kernel-size of that convolution (Default is 15).
    """
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "nr_channels" in kwargs, "Specify 'nr_channels' (int) to perturb the image."
    assert "img_size" in kwargs, "Specify 'img_size' (int) to perturb the image."
    assert (
        "indices" in kwargs
    ), "Specify 'indices' to enable perturbation function to run."

    # Get kwargs
    # The patch-size for the blur generation (NOT the patch-size for perturbation)
    blur_patch_size = kwargs.get("blur_patch_size", 15)
    nr_channels = kwargs.get("nr_channels", 3)
    img_size = kwargs.get("img_size", 224)

    # Reshape image since rest of function assumes channels_first
    img = img.reshape(nr_channels, img_size, img_size)
    kwargs["indices"] = (slice(0, kwargs.get("nr_channels")),) + kwargs["indices"]

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
    img_perturbed[kwargs["indices"]] = avgimg[kwargs["indices"]]

    return img_perturbed


def uniform_sampling(img: np.array, **kwargs) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 1D array."
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
                choice=kwargs["perturb_baseline"], img=img **kwargs
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
