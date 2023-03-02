"""This module contains a collection of norm functions i.e., ways to measure the norm of a input- (or explanation) vector."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations
from importlib import util
from functools import singledispatch
import numpy as np


@singledispatch
def fro_norm(a: np.array) -> float:
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
    return np.linalg.norm(a, ord=1, axis=0)


@singledispatch
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
    if a.ndim == 4:
        return np.linalg.norm(np.linalg.norm(a, axis=(-1, -2)), axis=-1)
    if a.ndim == 3:
        return np.linalg.norm(a, axis=(-1, -2))
    if a.ndim == 2:
        return np.linalg.norm(a, axis=-1)
    if a.ndim == 1:
        return np.linalg.norm(a)
    raise ValueError("This is unexpected.")


def linf_norm(a: np.array) -> float:
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
    return np.linalg.norm(a, ord=np.inf, axis=0)


if util.find_spec("tensorflow"):

    from quantus.helpers.utils import tf_function
    import tensorflow as tf

    @l2_norm.register(tf.Tensor)
    def _(a: tf.Tensor) -> tf.Tensor:
        ndim = len(tf.shape(a))
        if ndim == 4:
            return tf.linalg.norm(tf.linalg.norm(a, axis=(-1, -2)), axis=-1)
        if ndim == 3:
            return tf.linalg.norm(a, axis=(-1, -2))
        if ndim == 2:
            return tf.linalg.norm(a, axis=-1)
        if ndim == 1:
            return tf.linalg.norm(a)


    @fro_norm.register(tf.Tensor)
    @tf_function
    def _(a: tf.Tensor):
        return tf.linalg.norm(a, axis=0, ord="for")