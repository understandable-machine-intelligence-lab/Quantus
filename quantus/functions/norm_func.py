"""This module contains a collection of norm functions i.e., ways to measure the norm of a input- (or explanation) vector."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations
import numpy as np


def fro_norm(a: np.array) -> float | np.ndarray:
    """
    Calculate Frobenius norm for an array.

    Parameters
    ----------
    a: np.ndarray
         The array to calculate the Frobenius on.

    Returns
    -------
    float
        The norm.
    """
    return _ndim_norm(a, 1)


def l2_norm(a: np.array) -> float | np.ndarray:
    """
    Calculate L2 norm for an array.

    Parameters
    ----------
    a: np.ndarray
         The array to calculate the L2 on

    Returns
    -------
    float
        The norm.
    """

    return _ndim_norm(a, l_ord=2)


def linf_norm(a: np.array) -> float | np.ndarray:
    """
    Calculate L-inf norm for an array.

    Parameters
    ----------
    a: np.ndarray
         The array to calculate the L-inf on.

    Returns
    -------
    float
        The norm.
    """
    return _ndim_norm(a, l_ord=np.inf)


def _ndim_norm(a: np.ndarray, l_ord: int) -> float | np.ndarray:
    if a.ndim == 1:
        return np.linalg.norm(a, ord=l_ord)
    elif a.ndim == 2:
        return np.linalg.norm(a, axis=-1, ord=l_ord)
    elif a.ndim == 3:
        return np.linalg.norm(a, axis=(-1, -2), ord=l_ord)  # noqa
    if a.ndim == 4:
        return np.linalg.norm(
            np.linalg.norm(a, axis=(-1, -2), ord=l_ord), axis=-1, ord=l_ord
        )  # noqa
    else:
        raise ValueError(f"Supported are ndim up to 4, but found: {a.ndim}")
