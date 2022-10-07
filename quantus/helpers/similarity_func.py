"""This module holds a collection of similarity functions i.e., ways to measure the distance between two inputs (or explanations)."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
# Quantus project URL: https://github.com/understandable-machine-intelligence-lab/Quantus

from typing import Union
import scipy
from sklearn import metrics
import skimage
import numpy as np


def correlation_spearman(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Spearman rank of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.stats.spearmanr(a, b)[0]


def correlation_pearson(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Pearson correlation of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.stats.pearsonr(a, b)[0]


def correlation_kendall_tau(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Kendall Tau correlation of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.stats.kendalltau(a, b)[0]


def distance_euclidean(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Euclidean distance of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.spatial.distance.euclidean(u=a, v=b)


def distance_manhattan(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Manhattan distance of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.spatial.distance.cityblock(u=a, v=b)


def distance_chebyshev(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Chebyshev distance of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.spatial.distance.chebyshev(u=a, v=b)


def lipschitz_constant(
    a: np.array,
    b: np.array,
    c: Union[np.array, None],
    d: Union[np.array, None],
    **kwargs
) -> float:
    """
    Calculate non-negative local Lipschitz abs(||a-b||/||c-d||), where a,b can be f(x) or a(x) and c,d is x.

    For numerical stability, a small value is added to division.

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    c: np.ndarray
         The third array to use for similarity scoring.
    d: np.ndarray
         The fourth array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    eps = 1e-10

    d1 = kwargs.get("norm_numerator", distance_manhattan)
    d2 = kwargs.get("norm_denominator", distance_euclidean)

    if np.shape(a) == ():
        return float(abs(a - b) / (d2(c, d) + eps))
    else:
        return float(d1(a, b) / (d2(a=c, b=d) + eps))


def abs_difference(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate the absolute difference between two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return np.mean(abs(a - b))


def cosine(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Cosine of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return scipy.spatial.distance.cosine(u=a, v=b)


def ssim(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Structural Similarity Index Measure of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return skimage.metrics.structural_similarity(
        im1=a, im2=b, win_size=kwargs.get("win_size", None)
    )


def difference(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate the difference between two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float
        The similarity score.
    """
    return a - b
