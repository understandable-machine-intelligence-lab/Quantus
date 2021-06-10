"""Collection of localization functions i..e, ways to measure how an attribution differs from a given ground truth. """
import numpy as np


def correlation_spearman(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Spearman rank of two images (or explanations)."""
    return scipy.stats.spearmanr(a, b)[0]


LOCALIZATION_FUNCTIONS = {
    "gaussian_blur": gaussian_blur,
}
