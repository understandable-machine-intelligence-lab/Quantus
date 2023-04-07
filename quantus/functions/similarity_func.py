"""This module holds a collection of similarity functions i.e., ways to measure the distance between two inputs (or explanations)."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
# Quantus project URL: https://github.com/understandable-machine-intelligence-lab/Quantus

from __future__ import annotations

from functools import wraps
from typing import Union

import numpy as np
import scipy
import skimage


def vectorize_similarity(func):
    vectorized_func = np.vectorize(func, signature="(n),(n)->()", cache=True)

    @wraps(func)
    def wrapper(a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        def flatten_over_batch(arr):
            shape = np.shape(arr)
            return np.reshape(arr, (shape[0], -1))

        if np.ndim(a) != np.ndim(b):
            raise ValueError(
                f"a and b must have same shapes, but found, {a.shape = }, {b.shape = }"
            )

        if np.ndim(a) == 1:
            return func(a, b)

        if np.ndim(a) > 2:
            a = flatten_over_batch(a)
            b = flatten_over_batch(b)

        return vectorized_func(a, b)

    return wrapper


@vectorize_similarity
def correlation_spearman(a: np.ndarray, b: np.ndarray, **kwargs) -> np.ndarray:
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


@vectorize_similarity
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


def correlation_kendall_tau(a: np.array, b: np.array, **kwargs) -> np.ndarray | float:
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
    **kwargs,
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
    return np.abs(a - b)


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


@vectorize_similarity
def ssim(a: np.array, b: np.array, **kwargs) -> float | np.ndarray:
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
    max_val = np.max(np.abs(np.concatenate([a, b])))
    data_range = 1.0 if max_val <= 1.0 else 255.0

    return skimage.metrics.structural_similarity(
        im1=a,
        im2=b,
        win_size=kwargs.get("win_size"),
        data_range=data_range,
    )


def difference(a: np.array, b: np.array, **kwargs) -> float | np.ndarray:
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
