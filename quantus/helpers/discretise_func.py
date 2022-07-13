"""This module holds a collection of explanation discretisation functions i..e, methods to split continuous explanation
spaces into discrete counterparts."""
import numpy as np


def floating_points(a: np.array, **kwargs) -> float:
    """
    Rounds input to have n floating-points representation. Returns the hash values of the resulting array.
    a (np.array): Numpy array with shape (x,).
    kwargs: Keyword arguments (optional)
        n: Number of floating point digits.
    """
    n = kwargs.get("n", 2)
    discretized_arr = a.round(decimals=n)
    return hash(bytes(discretized_arr))


def sign(a: np.array, **kwargs) -> float:
    """
    Calculates element-wise signs of the array. Returns the hash values of the resulting array.
    a (np.array): Numpy array with shape (x,).
    kwargs: Keyword arguments (optional)
    """
    discretized_arr = np.sign(a)
    return hash(bytes(discretized_arr))


def top_n_sign(a: np.array, **kwargs) -> float:
    """
    Calculates top n element-wise signs of the array. Returns the hash values of the resulting array.
    a (np.array): Numpy array with shape (x,).
    kwargs: Keyword arguments (optional)
        n: Number of floating point digits.
    """
    n = kwargs.get("n", 5)
    discretized_arr = np.sign(a)[:n]
    return hash(bytes(discretized_arr))


def rank(a: np.array, **kwargs) -> float:
    """
    Calculates indices that would sort the array in order of importance. Returns the hash values of the resulting array.
    a (np.array): Numpy array with shape (x,).
    kwargs: Keyword arguments (optional)
    """
    """Calculates indices that would sort the array. Returns the hash values of the resulting array."""
    discretized_arr = np.argsort(a)[::-1]
    return hash(bytes(discretized_arr))
