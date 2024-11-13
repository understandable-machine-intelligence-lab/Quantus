"""This module holds a collection of similarity functions i.e., ways to measure the distance between two inputs (or explanations)."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
# Quantus project URL: https://github.com/understandable-machine-intelligence-lab/Quantus

from typing import Union, List

import numpy as np
import scipy
import skimage
import sys


def correlation_spearman(a: np.array, b: np.array, batched: bool = False, **kwargs) -> Union[float, np.array]:
    """
    Calculate Spearman rank of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    batched: bool
         True if arrays are batched. Arrays are expected to be 2D (B x F), where B is batch size and F is the number of features
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    Union[float, np.array]
        The similarity score or a batch of similarity scores.
    """
    if batched:
        assert len(a.shape) == 2 and len(b.shape) == 2, "Batched arrays must be 2D"
        # Spearman correlation is not calculated row-wise like pearson. Instead it is calculated between each
        # pair from BOTH a and b
        correlation = scipy.stats.spearmanr(a, b, axis=1)[0]
        # if a and b batch size is 1, scipy returns a float instead of an array
        if correlation.shape:
            correlation = correlation[: len(a), len(a) :]
            return np.diag(correlation)
        else:
            return np.array([correlation])
    return scipy.stats.spearmanr(a, b)[0]


def correlation_pearson(a: np.array, b: np.array, batched: bool = False, **kwargs) -> Union[float, np.array]:
    """
    Calculate Pearson correlation of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    batched: bool
         True if arrays are batched. Arrays are expected to be 2D (B x F), where B is batch size and F is the number of features
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    Union[float, np.array]
        The similarity score or a batch of similarity scores.
    """
    if batched:
        assert len(a.shape) == 2 and len(b.shape) == 2, "Batched arrays must be 2D"
        # No axis parameter in older versions
        if sys.version_info >= (3, 10):
            return scipy.stats.pearsonr(a, b, axis=1)[0]
        return np.array([scipy.stats.pearsonr(aa, bb)[0] for aa, bb in zip(a, b)])
    return scipy.stats.pearsonr(a, b)[0]


def correlation_kendall_tau(a: np.array, b: np.array, batched: bool = False, **kwargs) -> Union[float, np.array]:
    """
    Calculate Kendall Tau correlation of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    batched: bool
         True if arrays are batched. Arrays are expected to be 2D (B x F), where B is batch size and F is the number of features
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    Union[float, np.array]
        The similarity score or a batch of similarity scores.
    """
    if batched:
        assert len(a.shape) == 2 and len(b.shape) == 2, "Batched arrays must be 2D"
        # No support for axis currently, so just iterating over the batch
        return np.array([scipy.stats.kendalltau(a_i, b_i)[0] for a_i, b_i in zip(a, b)])
    return scipy.stats.kendalltau(a, b)[0]


def distance_euclidean(a: np.array, b: np.array, **kwargs) -> Union[float, np.array]:
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
    Union[float, np.array]
        The similarity score or a batch of similarity scores.
    """
    return ((a - b) ** 2).sum(axis=-1) ** 0.5


def distance_manhattan(a: np.array, b: np.array, **kwargs) -> Union[float, np.array]:
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
    Union[float, np.array]
        The similarity score or a batch of similarity scores.
    """
    return abs(a - b).sum(-1)


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
        return abs(a - b) / (d2(c, d) + eps)
    else:
        return d1(a, b) / (d2(a=c, b=d) + eps)


def abs_difference(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate the mean of the absolute differences between two images (or explanations).

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


def squared_difference(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate the sqaured differences between two images (or explanations).

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
    return np.sum((a - b) ** 2)


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


def ssim(a: np.array, b: np.array, batched: bool = False, **kwargs) -> Union[float, List[float]]:
    """
    Calculate Structural Similarity Index Measure of two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
         The first array to use for similarity scoring.
    b: np.ndarray
         The second array to use for similarity scoring.
    batched: bool
         Whether the arrays are batched.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    Union[float, List[float]]
        The similarity score, returns a list if batched.
    """

    def inner(aa: np.array, bb: np.array) -> float:
        max_point, min_point = np.max(np.concatenate([aa, bb])), np.min(np.concatenate([aa, bb]))
        data_range = float(np.abs(max_point - min_point))
        return skimage.metrics.structural_similarity(
            im1=aa, im2=bb, win_size=kwargs.get("win_size", None), data_range=data_range
        )

    if batched:
        return [inner(aa, bb) for aa, bb in zip(a, b)]
    return inner(a, b)


def difference(a: np.array, b: np.array, **kwargs) -> np.array:
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
    np.array
        The difference in each element.
    """
    return a - b
