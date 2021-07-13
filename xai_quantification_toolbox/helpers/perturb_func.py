""" Collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2
import random

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


def baseline_replacement_by_indices(img: np.array, **kwargs) -> np.array:
    """Replace indices in an image by given baseline."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    assert "index" in kwargs, "Specify an 'index' to enable perturbation function to run."
    assert "perturb_baseline" in kwargs, "Specify a 'perturb_baseline' e.g., 0.0 for a black pixel replacement."
    img[kwargs.get("index", 0)] = kwargs.get("perturb_baseline", 0.0)
    return img


def baseline_replacement_by_patch(img: np.array, **kwargs) -> np.array:
    """Replace a single patch in an image by given baseline."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    assert "patch_size" in kwargs, "Specify a 'patch_size' (int) to enable perturbation function to run."
    assert "nr_channels" in kwargs, "Specify a 'nr_channels' (int) to enable perturbation function to run."
    assert "nr_channels" in kwargs, "Specify a 'perturb_baseline' (int) to enable perturbation function to run."
    assert "top_left_y" in kwargs, "Specify a 'top_left_y' (int) to enable perturbation function to run."
    assert "top_left_x" in kwargs, "Specify a 'top_left_x' (int) to enable perturbation function to run."

    def _set_baseline(**kwargs) -> float:
        """Set baseline based on string input."""
        available_baselines = ["random", "black", "white", "mean", "neighbourhood"]
        assert kwargs["perturb_baseline"] in available_baselines, f"Specify a perturb_baseline (str) that exist in {available_baselines}"

        # Mask input with masking value.
        if kwargs["perturb_baseline"].lower() == "random":
            return float(random.random())
        elif kwargs["perturb_baseline"].lower() == "black":
            return 0.0
        elif kwargs["perturb_baseline"].lower() == "white":
            return 1.0
        elif kwargs["perturb_baseline"].lower() == "mean":
            return float(kwargs["patch"].mean())
        elif kwargs["perturb_baseline"].lower() == "neighbourhood":
            return float(random.uniform(kwargs["patch"].min(), kwargs["patch"].max()))
        else:
            ValueError("Specify a perturb_baseline (str) that exist in {}.").__format__(available_baselines)

    #for c in range(kwargs.get("nr_channels", 3)):
    if not isinstance(kwargs.get("perturb_baseline", 0.0), (float, int)):
        kwargs["patch"] = img[:, kwargs["top_left_x"]: kwargs["top_left_x"] + kwargs["patch_size"], kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"]]
        kwargs["perturb_baseline"] = _set_baseline(**kwargs)

    img[:, kwargs["top_left_x"] : kwargs["top_left_x"] + kwargs["patch_size"], kwargs["top_left_y"] : kwargs["top_left_y"] + kwargs["patch_size"]] = kwargs["perturb_baseline"]

    return img


def uniform_sampling(img: np.array, **kwargs) -> np.array:
    """Add noise to input as sampled uniformly random from L_infiniy ball with a radius."""
    assert img.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return img + np.random.uniform(-kwargs.get("perturb_radius", 0.02), kwargs.get("perturb_radius", 0.02), size=img.shape)


def rotation(img: np.array, **kwargs) -> np.array:
    """Rotate image by some given angle."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
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
    matrix = np.float32([[1, 0, kwargs.get("perturb_dx", 10)], [0, 1, 0]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (224, 224),
            borderValue=kwargs.get("perturb_baseline", 0.75),
        ),
        2,
        0,
    )


def translation_y_direction(img: np.array, **kwargs) -> np.array:
    """Translate image by some given value in the x-direction."""
    assert img.ndim == 3, "Check that 'perturb_func' receives a 3D array."
    matrix = np.float32([[1, 0, 0], [0, 1, kwargs.get("perturb_dy", 10)]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (224, 224),
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


PERTURBATION_FUNCTIONS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "baseline_replacement_by_indices": baseline_replacement_by_indices,
    "baseline_replacement_by_patch": baseline_replacement_by_patch,
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
    "optimization_scheme": optimization_scheme,
    "uniform_sampling": uniform_sampling
}
