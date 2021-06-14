""" Collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2


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
        kwargs.get("mean", 0.0), kwargs.get("perturb_std", 0.1), size=img.size
    )


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
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
    "optimization_scheme": optimization_scheme,
    "uniform_sampling": uniform_sampling
}
