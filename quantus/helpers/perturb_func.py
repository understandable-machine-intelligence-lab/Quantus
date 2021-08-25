""" Collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2
import random

# TODO. Rewrite to help user like here: https://captum.ai/api/_modules/captum/metrics/_core/infidelity.html#infidelity.
"""
def perturb_func (callable):
    The perturbation function of model inputs. This function takes
    model inputs and optionally baselines as input arguments and returns
    either a tuple of perturbations and perturbed inputs or just
    perturbed inputs. For example:

    >>> def my_perturb_func(inputs):
    >>>   <MY-LOGIC-HERE>
    >>>   return perturbations, perturbed_inputs

"""

def gaussian_blur(img: np.array, **kwargs) -> np.array:
    """Inject gaussian blur to the input. """
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return scipy.ndimage.gaussian_filter(
        img, sigma=kwargs.get("perturb_sigma", 0.1) * np.max(img)
    )


def gaussian_noise(img: np.array, **kwargs) -> np.array:
    """Add gaussian noise to the input. """
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return img + np.random.normal(
        kwargs.get("mean", 0.0), kwargs.get("perturb_std", 0.01), size=img.size
    )


def get_baseline(img: np.array, **kwargs) -> float:
    """Get baseline based on a uer-defined string or integer input.
    TODO. Remove hardcoded dictionary and allow user to flexibly specify its own replacement."""
    assert (("perturb_baseline" in kwargs) or ("replacement_values" in kwargs)), "Specify a 'perturb_baseline' \
    e.g., 0.0 or 'black' for pixel replacement or 'replacement_values' containing an array with one value per \
    index for replacement."

    if "replacement_values" in kwargs:
        return kwargs["replacement_values"]

    elif isinstance(kwargs.get("perturb_baseline", None), (float, int)):
        return kwargs["perturb_baseline"]

    else:
        mask_dict = {
            "random": float(random.random()),
            "uniform": float(random.uniform(img.min(), img.max())),
            "black": float(img.min()),
            "white": float(img.max()),
        }

        if "patch" in kwargs:
            mask_dict["neighbourhood_mean"] = float(kwargs["patch"].mean()),
            mask_dict["neighbourhood_random_min_max"] = float(random.uniform(kwargs["patch"].min(), kwargs["patch"].max())),

        assert kwargs["perturb_baseline"] in list(
            mask_dict.keys()
        ), f"Specify 'perturb_baseline' (str) that exist in {list(mask_dict.keys())}"

        try:
            return mask_dict[kwargs["perturb_baseline"].lower()]
        except:
            return 0.0


def baseline_replacement_by_indices(img: np.array, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    assert (
        "index" in kwargs
    ), "Specify 'index' to enable perturbation function to run."

    img[kwargs["index"]] = get_baseline(img, **kwargs)
    return img


def baseline_replacement_by_patch(img: np.array, **kwargs) -> np.array:
    """Replace a single patch in an image by given baseline."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert (
        "patch_size" in kwargs
    ), "Specify 'patch_size' (int) to perturb the image."
    assert (
        "nr_channels" in kwargs
    ), "Specify 'nr_channels' (int) to perturb the image."
    assert (
        "nr_channels" in kwargs
    ), "Specify 'perturb_baseline' (int, float, str) to perturb the image."
    assert (
        "top_left_y" in kwargs
    ), "Specify 'top_left_y' (int) to perturb the image."
    assert (
        "top_left_x" in kwargs
    ), "Specify 'top_left_x' (int) to perturb the image."

    # Preset patch for 'mean' and 'neighbourhood' choices.
    kwargs["patch"] = img[:, kwargs["top_left_x"]: kwargs["top_left_x"] + kwargs["patch_size"],
                      kwargs["top_left_y"]: kwargs["top_left_y"] + kwargs["patch_size"]]

    # for c in range(kwargs.get("nr_channels", 3)):
    img[
        :,
        kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"],
        kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"],
    ] = get_baseline(img, **kwargs)

    return img


def uniform_sampling(img: np.array, **kwargs) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return img + np.random.uniform(
        -kwargs.get("perturb_radius", 0.02),
        kwargs.get("perturb_radius", 0.02),
        size=img.shape,
    )


def rotation(img: np.array, **kwargs) -> np.array:
    """Rotate image by some given angle."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = cv2.getRotationMatrix2D(
        center=(kwargs.get("img_size", 224) / 2, kwargs.get("img_size", 224) / 2),
        angle=kwargs.get("perturb_angle", 10),
        scale=1,
    )
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
        ),
        2,
        0,
    )


def translation_x_direction(img: np.array, **kwargs) -> np.array:
    """Translate image by some given value in the x-direction."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = np.float32([[1, 0, kwargs.get("perturb_dx", 10)], [0, 1, 0]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            borderValue=kwargs.get("perturb_baseline", 0.75),
        ),
        2,
        0,
    )


def translation_y_direction(img: np.array, **kwargs) -> np.array:
    """Translate image by some given value in the x-direction."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "img_size" in kwargs, "Specify 'img_size' to perform translation."
    matrix = np.float32([[1, 0, 0], [0, 1, kwargs.get("perturb_dy", 10)]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (kwargs.get("img_size", 224), kwargs.get("img_size", 224)),
            borderValue=kwargs.get("perturb_baseline", 0.75),
        ),
        2,
        0,
    )


def optimization_scheme(img: np.array, **kwargs) -> np.array:
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    # https://github.com/google-research-datasets/bam/blob/master/scripts/construct_delta_patch.py
    # Use gradient descent to optimize for a perturbation that modifies those pixels
    # within a small L2 distance of initialization ...
    # TODO. Implement optimization_scheme
    return gaussian_noise(img, **kwargs)

