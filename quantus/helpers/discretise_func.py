"""This module holds a collection of explanation discretisation functions i.e., methods to split continuous explanation
spaces into discrete counterparts."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import numpy as np


def floating_points(a: np.array, **kwargs) -> float:
    """
    Rounds input to have n floating-points representation

    Parameters
    ----------
    a: np.ndarray
         Numpy array with shape (x,).
    kwargs: optional
            Keyword arguments.
        n: integer
        Number of floating point digits.

    Returns
    -------
    float
        Returns the hash values of the resulting array.
    """
    n = kwargs.get("n", 2)
    discretized_arr = a.round(decimals=n)
    return hash(bytes(discretized_arr))


def sign(a: np.array, **kwargs) -> float:
    """
    Calculates element-wise signs of the array.

    Parameters
    ----------
    a: np.ndarray
         Numpy array with shape (x,).
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    float
        Returns the hash values of the resulting array.
    """
    discretized_arr = np.sign(a)
    return hash(bytes(discretized_arr))


def top_n_sign(a: np.array, **kwargs) -> float:
    """
    Calculates top n element-wise signs of the array.

    Parameters
    ----------
    a: np.ndarray
         Numpy array with shape (x,).
    kwargs: optional
            Keyword arguments.
        n: integer
        Number of floating point digits.

    Returns
    -------
    float
        Returns the hash values of the resulting array.
    """
    n = kwargs.get("n", 5)
    discretized_arr = np.sign(a)[:n]
    return hash(bytes(discretized_arr))


def rank(a: np.array, **kwargs) -> float:
    """
    Calculates indices that would sort the array in order of importance.

    Parameters
    ----------
    a: np.ndarray
         Numpy array with shape (x,).
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    float
        Returns the hash values of the resulting array.
    """
    discretized_arr = np.argsort(a)[::-1]
    return hash(bytes(discretized_arr))
