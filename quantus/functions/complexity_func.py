"""This module holds a collection of loss functions i.e., ways to measure the loss between two inputs."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import scipy
import numpy as np


def entropy(a: np.array, x: np.array, **kwargs) -> float:
    """
    Calculate entropy.

    Parameters
    ----------
    a: np.ndarray
        Array to calculate entropy on.
    x: np.ndarray
        Array to compute shape.
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    float:
        A floating point, raning [0, inf].
    """
    assert (a >= 0).all(), "Entropy computation requires non-negative attributions"

    if len(x.shape) == 1:
        newshape = np.prod(x.shape)
    else:
        newshape = np.prod(x.shape[1:])

    a_reshaped = np.reshape(a, int(newshape))
    a_normalised = a_reshaped.astype(np.float64) / np.sum(np.abs(a_reshaped))
    return scipy.stats.entropy(pk=a_normalised)

def gini_coeffiient(a: np.array, x: np.array, **kwargs) -> float:
    """
    Calculate Gini coefficient.

    Parameters
    ----------
    a: np.ndarray
        Array to calculate gini_coeffiient on.
    x: np.ndarray
        Array to compute shape.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    float:
        A floating point, ranging [0, 1].

    """

    if len(x.shape) == 1:
        newshape = np.prod(x.shape)
    else:
        newshape = np.prod(x.shape[1:])

    a = np.array(np.reshape(a, newshape), dtype=np.float64)
    a += 0.0000001
    a = np.sort(a)
    score = (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a)) / (
        a.shape[0] * np.sum(a)
    )
    return score

def discrete_entropy(a: np.array, x: np.array, **kwargs) -> float:
    """
    Calculate discrete entropy of explanations with n_bins equidistant spaced bins
    Parameters
    ----------
    a: np.ndarray
        Array to calculate entropy on.
    x: np.ndarray
        Array to compute shape.
    kwargs: optional
        Keyword arguments.

        n_bins: int
            Number of bins. default is 100.

    Returns
    -------
    float:
        Discrete Entropy.
    """

    n_bins = kwargs.get("n_bins", 100)

    histogram, bins = np.histogram(a, bins=n_bins)

    return scipy.stats.entropy(pk=histogram)

def freedman_diaconis_rule(a_batch: np.array) -> int:
    """Freedman–Diaconis' rule."""

    iqr = np.percentile(a_batch, 75) - np.percentile(a_batch, 25)
    n = a_batch[0].ndim
    bin_width = 2 * iqr / np.power(n, 1/3)

    # Set a minimum value for bin_width to avoid division by very small numbers.
    min_bin_width = 1e-6
    bin_width = max(bin_width, min_bin_width)

    # Calculate number of bins based on bin width.
    n_bins = int((np.max(a_batch) - np.min(a_batch)) / bin_width)

    return n_bins

def scotts_rule(a_batch: np.array) -> int:
    """Scott's rule."""

    std = np.std(a_batch)
    n = a_batch[0].ndim

    # Calculate bin width using Scott's rule.
    bin_width = 3.5 * std / np.power(n, 1/3)

    # Calculate number of bins based on bin width.
    n_bins = int((np.max(a_batch) - np.min(a_batch)) / bin_width)

    return n_bins

import numpy as np

def square_root_choice(a_batch: np.array) -> int:
    """Square-root choice rule."""

    n = a_batch[0].ndim
    n_bins = int(np.sqrt(n))

    return n_bins

def sturges_formula(a_batch: np.array) -> int:
    """Sturges' formula."""

    n = a_batch[0].ndim
    n_bins = int(np.log2(n) + 1)

    return n_bins

def rice_rule(a_batch: np.array) -> int:
    """Rice Rule."""

    n = a_batch[0].ndim
    n_bins = int(2 * np.power(n, 1/3))

    return n_bins
