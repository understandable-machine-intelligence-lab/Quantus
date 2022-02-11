"""This module provides some basic functionality to normalise and denormalise images."""
from typing import Union
import numpy as np


def normalise_by_max(a: np.ndarray) -> np.ndarray:
    """ "Normalize attributions by the maximum absolute value of the explanation."""
    a /= np.max(np.abs(a))
    return a


def normalise_by_negative(a: np.ndarray) -> np.ndarray:
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""
    if a.min() >= 0.0:
        return a / a.max()
    if a.max() <= 0.0:
        return -a / a.min()
    return (a > 0.0) * a / a.max() - (a < 0.0) * a / a.min()


def denormalise(
    img: np.ndarray,
    mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
    std: np.ndarray = np.array([0.229, 0.224, 0.225]),
    **kwargs
) -> np.ndarray:
    """De-normalise a torch image (using conventional ImageNet values)."""
    return (np.array(img) * std.reshape(-1, 1, 1)) + mean.reshape(-1, 1, 1)
