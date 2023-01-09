"""This module holds a collection of loss functions i.e., ways to measure the loss between two inputs."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import numpy as np


def mse(a: np.array, b: np.array, **kwargs) -> float:
    """
    Calculate Mean Squared Error between two images (or explanations).

    Parameters
    ----------
    a: np.ndarray
             Array to calculate MSE with.
    b: np.ndarray
             Array to calculate MSE with.
    kwargs: optional
            Keyword arguments.
        normalise_mse: boolean
            Indicates whether to returned a normalised MSE calculation or not.

    Returns
    -------
    float:
        A floating point of MSE.
    """

    normalise = kwargs.get("normalise_mse", False)

    if normalise:
        # Calculate MSE in its polynomial expansion (a-b)^2 = a^2 - 2ab + b^2.
        return np.average(((a**2) - (2 * (a * b)) + (b**2)), axis=0)
    # If no need to normalise, return (a-b)^2.

    return np.average(((a - b) ** 2), axis=0)
