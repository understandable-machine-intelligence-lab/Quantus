"""This module provides some basic functionality to normalise and denormalise images."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from typing import Optional, Sequence

import numpy as np


def normalise_by_max(
    a: np.ndarray,
    normalise_axes: Optional[Sequence[int]] = None,
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
    a: np.ndarray,
    normalise_axes: Optional[Sequence[int]] = None,
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
    # TODO: Temporary solution to catch an elusive bug causing a numpy RuntimeWarning below.
    #       Will be removed once bug is fixed.
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            return_array = np.where(
                np.logical_and(a_min < 0.0, a_max > 0.0),
                (a > 0.0) * np.divide(a, a_max, where=a_max != 0)
                - (a < 0.0) * np.divide(a, a_min, where=a_min != 0),
                return_array,
            )
        except RuntimeWarning:
            print(
                "Encountered a RuntimeWarning in numpy operation during normalise_by_negative, although both nan and inf values should be impossible here."
                "If this occurred, please try to use a different normalisation function, e.g., normalising by maximum of unsigned array."
            )
            print(
                "-----------------------------------------------------DEBUG OUTPUT------------------------------------------------------"
            )
            print("a_max: {}".format(a_max))
            print("a_min: {}".format(a_min))
            print(
                "np.logical_and(a_min < 0.0, a_max > 0.0): {}".format(
                    np.logical_and(a_min < 0.0, a_max > 0.0)
                )
            )
            print("(a > 0.0): {}".format((a > 0.0)))
            print(
                "np.divide(a, a_max, where=a_max != 0): {}".format(
                    np.divide(a, a_max, where=a_max != 0)
                )
            )
            print(
                "(a > 0.0) * np.divide(a, a_max, where=a_max != 0): {}".format(
                    (a > 0.0) * np.divide(a, a_max, where=a_max != 0)
                )
            )
            print("(a < 0.0): {}".format((a > 0.0)))
            print(
                "np.divide(a, a_min, where=a_min != 0): {}".format(
                    np.divide(a, a_min, where=a_min != 0)
                )
            )
            print(
                "(a < 0.0) * np.divide(a, a_min, where=a_min != 0): {}".format(
                    (a < 0.0) * np.divide(a, a_min, where=a_min != 0)
                )
            )
            raise

    return return_array


def denormalise(
    a: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
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


def normalise_by_average_second_moment_estimate(
    a: np.ndarray,
    normalise_axes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Normalise attributions by dividing the attribution map by the square-root
    of its average second moment estimate (that is, similar to the standard
    deviation, but centered around zero instead of the data mean).

    This normalisation function does not normalise the attributions into a fixed range.
    Instead, it ensures that each score in the attribution map has an average squared
    distance to zero that is equal to one. This is not meant for visualisation purposes,
    rather it is meant to preserve a quantity that is useful for the comparison of distances
    between different attribution methods.

    References:
        1) Binder et al., (2022): "Shortcomings of Top-Down Randomization-Based Sanity Checks
        for Evaluations of Deep Neural Network Explanations." arXiv: https://arxiv.org/abs/2211.12486.

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

    # Check that square root of the second momment estimatte is nonzero.
    second_moment_sqrt = np.sqrt(
        np.sum(a**2, axis=normalise_axes, keepdims=True)
        / np.prod([a.shape[n] for n in normalise_axes])
    )

    if all(second_moment_sqrt != 0):
        a /= np.sqrt(
            np.sum(a**2, axis=normalise_axes, keepdims=True)
            / np.prod([a.shape[n] for n in normalise_axes])
        )
    else:
        warnings.warn(
            "Encountered second moment of parameter 'a' equal to zero "
            "in normalise_by_average_second_moment_estimate. As a result, no normalisation is performed. "
            "Be aware that this may cause inconsistencies in your results."
        )

    return a
