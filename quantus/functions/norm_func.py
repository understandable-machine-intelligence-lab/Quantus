"""This module contains a collection of norm functions i.e., ways to measure the norm of a input- (or explanation) vector."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations


from functools import wraps
from typing import Callable

import numpy as np


def vectorize_norm(
    func: Callable[[np.ndarray], float]
) -> Callable[[np.ndarray], np.ndarray | float]:
    """Decorator, which allow calling norm_functions on 1D and 2D inputs."""
    vectorized_func = np.vectorize(func, signature="(n)->()", cache=True)

    @wraps(func)
    def wrapper(a):
        a = np.asarray(a)
        ndim = np.ndim(a)
        if ndim == 1:
            return func(a)
        if ndim > 2:
            raise ValueError(
                f"{func.__name__} supports only 1d and 2D inputs, but {a.ndim = }."
            )

        return vectorized_func(a)

    return wrapper


@vectorize_norm
def fro_norm(a: np.ndarray) -> float:
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
    assert a.ndim == 1, "Check that 'fro_norm' receives a 1D array."
    return np.linalg.norm(a)


@vectorize_norm
def l2_norm(a: np.ndarray) -> float:
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
    assert a.ndim == 1, "Check that 'l2_norm' receives a 1D array."
    return np.linalg.norm(a)


@vectorize_norm
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
    assert a.ndim == 1, "Check that 'linf_norm' receives a 1D array."
    return np.linalg.norm(a, ord=np.inf)
