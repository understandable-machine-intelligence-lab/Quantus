"""This module provides some basic functionality to normalise and denormalise images."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Callable, Dict, Optional, Union, Sequence
import numpy as np


def normalise_by_max(
    a: np.ndarray, normalise_axes: Optional[Sequence[int]] = None, **kwargs
) -> np.ndarray:
    """
    Normalise attributions by the maximum absolute value of the explanation.

    Parameters
    ----------
    a: np.ndarray
         the array to normalise, e.g., an image or an explanation.
    normalise_axes: optional, sequence
        the axes to normalise over.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    a: np.ndarray
         a normalised array.
    """

    # No normalisation if a is only zeros.
    if np.all(a == 0.0):
        return a

    # Default normalise_axes.
    if normalise_axes is None:
        normalise_axes = list(range(np.ndim(a)))

    # Cast Sequence to tuple so numpy accepts it.
    normalise_axes = tuple(normalise_axes)

    a_max = np.max(np.abs(a), axis=normalise_axes, keepdims=True)
    a = np.divide(a, a_max)
    return a


def normalise_by_negative(
    a: np.ndarray, normalise_axes: Optional[Sequence[int]] = None, **kwargs
) -> np.ndarray:
    """
    Normalise attributions between [-1, 1].

    Parameters
    ----------
    a: np.ndarray
         the array to normalise, e.g., an image or an explanation.
    normalise_axes: optional, sequence
            the axes to normalise over.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    a: np.ndarray
         a normalised array.
    """

    # No normalisation if a is only zeros.
    if np.all(a == 0.0):
        return a

    # Default normalise_axes.
    if normalise_axes is None:
        normalise_axes = list(range(np.ndim(a)))

    # Cast Sequence to tuple so numpy accepts it.
    normalise_axes = tuple(normalise_axes)

    # Build return array from three cases.
    return_array = np.zeros(a.shape, dtype=a.dtype)

    # Calculate max and min values.
    a_max = a.max(axis=normalise_axes, keepdims=True)
    a_min = a.min(axis=normalise_axes, keepdims=True)

    # Case a.min() >= 0.0.
    return_array = np.where(
        a_min >= 0.0,
        np.divide(a, a_max, where=a_max != 0),
        return_array,
    )

    # Case a.max() <= 0.0.
    return_array = np.where(
        a_max <= 0.0,
        -np.divide(a, a_min, where=a_min != 0),
        return_array,
    )

    # Else.
    return_array = np.where(
        np.logical_and(a_min < 0.0, a_max > 0.0),
        (a > 0.0) * np.divide(a, a_max, where=a_max != 0)
        - (a < 0.0) * np.divide(a, a_min, where=a_min != 0),
        return_array,
    )

    return return_array


def denormalise(
    a: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """

    Parameters
    ----------
    a: np.ndarray
         the array to normalise, e.g., an image or an explanation.
    mean: np.ndarray
         The mean points to sample from, len(mean) = nr_channels.
    std: np.ndarray
         The standard deviations to sample from, len(mean) = nr_channels.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    np.ndarray
        A denormalised array.
    """
    return (np.array(a) * std.reshape(-1, 1, 1)) + mean.reshape(-1, 1, 1)
