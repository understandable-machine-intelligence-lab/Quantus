"""This module contains a collection of norm functions i.e., ways to measure the norm of a input- (or explanation) vector."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

import numpy as np


def get_norm_axis(arr):
    ndim = np.ndim(arr)
    if ndim == 1:
        return None
    if ndim == 2:
        return 1
    raise ValueError(
        f"Norm functions expect 1D or 2D inputs, but found: {np.ndim(arr)}"
    )


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
    # As per numpy docs
    # =====  ============================  ==========================
    # ord    norm for matrices             norm for vectors
    # =====  ============================  ==========================
    # None   Frobenius norm                2-norm
    # 'fro'  Frobenius norm                --
    # =====  ============================  ==========================
    # Frobenius norm is defined only for matrices.

    ndim = np.ndim(a)
    if ndim == 2:
        axis = None
    elif ndim == 3:
        axis = 0
    else:
        raise ValueError(
            f"Frobenius norm is defined only for matrices, so expected array to be 2D or 3D, but found: {ndim}D"
        )
    return np.linalg.norm(a, axis=axis, ord="fro")


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
    # As per numpy docs
    # =====  ============================  ==========================
    # ord    norm for matrices             norm for vectors
    # =====  ============================  ==========================
    # None   Frobenius norm                2-norm
    # =====  ============================  ==========================
    # We ensure we have a batch of flat vectors, or just one vector.
    axis = get_norm_axis(a)
    return np.linalg.norm(a, axis=axis)


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
    axis = get_norm_axis(a)
    return np.linalg.norm(a, ord=np.inf, axis=axis)
