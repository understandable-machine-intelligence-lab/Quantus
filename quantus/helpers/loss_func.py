"""This module holds a collection of loss functions i.e., ways to measure the loss between two inputs."""
import numpy as np


def mse(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Mean Squared Error between two images (or explanations)."""
    normalise = kwargs.get("normalise", False)
    if normalise:
        # Calculate MSE in its polynomial expansion (a-b)^2 = a^2 - 2ab + b^2.
        return np.average(((a**2) - (2 * (a * b)) + (b**2)), axis=0)
    # If no need to normalise, return (a-b)^2.
    return np.average(((a - b) ** 2), axis=0)
