"""This module holds a collection of similarity functions i..e, ways to measure the distance between two inputs (or explanations)."""
from typing import Union
import scipy
from sklearn import metrics
import skimage
import numpy as np


def correlation_spearman(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Spearman rank of two images (or explanations)."""
    return scipy.stats.spearmanr(a, b)[0]


def correlation_pearson(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Pearson correlation of two images (or explanations)."""
    return scipy.stats.pearsonr(a, b)[0]


def correlation_kendall_tau(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Kendall Tau correlation of two images (or explanations)."""
    return scipy.stats.kendalltau(a, b)[0]


def distance_euclidean(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Euclidean distance of two images (or explanations)."""
    return scipy.spatial.distance.euclidean(u=a, v=b)


def distance_manhattan(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Manhattan distance of two images (or explanations)."""
    return scipy.spatial.distance.cityblock(u=a, v=b)


def distance_chebyshev(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Chebyshev distance of two images (or explanations)."""
    return scipy.spatial.distance.chebyshev(u=a, v=b)


def lipschitz_constant(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate non-negative local Lipschitz abs(||a-b||/||c-d||), where a,b can be f(x) or a(x) and c,d is x."""

    d1 = kwargs.get("distance_numerator", distance_manhattan)
    d2 = kwargs.get("distance_denominator", distance_euclidean)

    if np.shape(a) == ():
        return float(abs(a - b) / d2(c, d))
    else:
        return float(d1(a, b) / d2(a=c, b=d))


def abs_difference(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate the absolute difference between two images (or explanations)."""
    return np.mean(abs(a - b))


def cosine(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Cosine of two images (or explanations)."""
    return scipy.spatial.distance.cosine(u=a, v=b)


def ssim(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Structural Similarity Index Measure of two images (or explanations)."""
    return skimage.metrics.structural_similarity(
        im1=a, im2=b, win_size=kwargs.get("win_size", None)
    )


def mse(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Mean Squared Error between two images (or explanations)."""
    return metrics.mean_squared_error(y_true=a, y_pred=b)


def difference(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate the difference between two images (or explanations)."""
    return a - b
