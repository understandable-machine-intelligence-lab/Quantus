""" Collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import numpy as np
import scipy
import cv2


def gaussian_blur(img: np.array, **kwargs):
    """Inject gaussian blur to the input. """
    return scipy.ndimage.gaussian_filter(
        img, sigma=kwargs.get("sigma", 0.1) * np.max(img)
    )


def gaussian_noise(img: np.array, **kwargs):
    """Add gaussian noise to the input. """
    return img + np.random.normal(
        kwargs.get("mean", 0.0), kwargs.get("std", 0.1), img.shape
    )


def rotation(img: np.array, **kwargs):
    """Rotate image by some given angle."""
    matrix = cv2.getRotationMatrix2D(
        center=(kwargs.get("img_size", 224) / 2, kwargs.get("img_size", 224) / 2),
        angle=kwargs.get("angle", 10),
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


def translation_x_direction(img: np.array, **kwargs):
    """Translate image by some given value in the x-direction."""
    matrix = np.float32([[1, 0, kwargs.get("dx", 10)], [0, 1, 0]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (224, 224),
            borderValue=kwargs.get("baseline_value", 0.75),
        ),
        2,
        0,
    )


def translation_y_direction(img: np.array, **kwargs):
    """Translate image by some given value in the x-direction."""
    matrix = np.float32([[1, 0, 0], [0, 1, kwargs.get("dy", 10)]])
    return np.moveaxis(
        cv2.warpAffine(
            np.moveaxis(img, 0, 2),
            matrix,
            (224, 224),
            borderValue=kwargs.get("baseline_value", 0.75),
        ),
        2,
        0,
    )


PERTURBATION_FUNCTIONS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
}
