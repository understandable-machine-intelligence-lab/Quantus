""" Collection of similarity functions i..e, ways to measure the distance between two inputs (or explanations). """
from typing import Union
import scipy
import sklearn
import skimage
import numpy as np


def correlation_spearman(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Spearman rank of two images (or explanations)."""
    return scipy.stats.spearmanr(a, b)[0]


def correlation_pearson(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Pearson correlation of two images (or explanations)."""
    return scipy.stats.pearsonr(a, b)[0]


def correlation_kendall_tau(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Kendall Tau correlation of two images (or explanations)."""
    return scipy.stats.kendalltau(a, b)[0]


def distance_euclidean(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Euclidean distance of two images (or explanations)."""
    return scipy.spatial.distance.euclidean(u=a, w=b)


def distance_manhattan(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Manhattan distance of two images (or explanations)."""
    return scipy.spatial.distance.cityblock(u=a, w=b)


def distance_chebyshev(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Chebyshev distance of two images (or explanations)."""
    return scipy.spatial.distance.chebyshev(u=a, w=b)


def cosine(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Cosine of two images (or explanations)."""
    return scipy.spatial.distance.cosine(u=a, w=b)


def ssim(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Structural Similarity Index Measure of two images (or explanations)."""
    return skimage.metrics.structural_similarity(
        im1=a, im2=b, win_size=kwargs.get("win_size", None)
    )


def mse(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate Mean Squared Error between two images (or explanations)."""
    return sklearn.metrics.mean_squared_error(y_true=a, y_pred=b)


def lipschitz_constant(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """Calculate non-negative local Lipschitz abs(||a-b||/||c-d||), where a,b can be f(x) or a(x) and c,d is x."""
    if np.shape(a) == ():
        return float(abs(a - b) / kwargs.get("norm", distance_euclidean)(c, d))
    else:
        return float(
            kwargs.get("norm", distance_euclidean)(a - b)
            / kwargs.get("norm", distance_euclidean)(c, d)
        )
