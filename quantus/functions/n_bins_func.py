"""This module holds a collection of algorithms to calculate a number of bins to use for entropy calculation."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import scipy
import numpy as np


def freedman_diaconis_rule(a_batch: np.ndarray) -> int:
    """
    Freedman–Diaconis' rule to compute the number of bins.

    Parameters
    ----------
    a_batch: np.ndarray
        The batch of attributions to use in the calculation.

    Returns
    -------
        integer
    """

    iqr = np.percentile(a_batch, 75) - np.percentile(a_batch, 25)
    n = a_batch[0].ndim
    bin_width = 2 * iqr / np.power(n, 1 / 3)

    # Set a minimum value for bin_width to avoid division by very small numbers.
    min_bin_width = 1e-6
    bin_width = max(bin_width, min_bin_width)

    # Calculate number of bins based on bin width.
    n_bins = int((np.max(a_batch) - np.min(a_batch)) / bin_width)

    return n_bins


def scotts_rule(a_batch: np.ndarray) -> int:
    """
    Scott's rule to compute the number of bins.

    Parameters
    ----------
    a_batch: np.ndarray
        The batch of attributions to use in the calculation.

    Returns
    -------
        integer
    """

    std = np.std(a_batch)
    n = a_batch[0].ndim

    # Calculate bin width using Scott's rule.
    bin_width = 3.5 * std / np.power(n, 1 / 3)

    # Calculate number of bins based on bin width.
    n_bins = int((np.max(a_batch) - np.min(a_batch)) / bin_width)

    return n_bins


def square_root_choice(a_batch: np.ndarray) -> int:
    """
    Square-root choice rule to compute the number of bins.

    Parameters
    ----------
    a_batch: np.ndarray
        The batch of attributions to use in the calculation.

    Returns
    -------
        integer
    """

    n = a_batch[0].ndim
    n_bins = int(np.sqrt(n))

    return n_bins


def sturges_formula(a_batch: np.ndarray) -> int:
    """
    Sturges' rule to compute the number of bins.

    Parameters
    ----------
    a_batch: np.ndarray
        The batch of attributions to use in the calculation.

    Returns
    -------
        integer
    """

    n = a_batch[0].ndim
    n_bins = int(np.log2(n) + 1)

    return n_bins


def rice_rule(a_batch: np.ndarray) -> int:
    """
    Rice rule to compute the number of bins.

    Parameters
    ----------
    a_batch: np.ndarray
        The batch of attributions to use in the calculation.

    Returns
    -------
        integer
    """

    n = a_batch[0].ndim
    n_bins = int(2 * np.power(n, 1 / 3))

    return n_bins
