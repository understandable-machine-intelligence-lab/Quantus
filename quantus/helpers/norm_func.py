"""This module contains a collection of norm functions i.e., ways to measure the norm of a input- (or explanation) vector."""

import numpy as np


def fro_norm(a: np.array) -> float:
    """
    Calculate Frobenius norm for an array.

    Parameters
    ----------
        a (np.array): the array to calculate the Frobenius on.

    Returns
    -------
        (float): the norm.
    """
    assert a.ndim == 1, "Check that 'fro_norm' receives a 1D array."
    return np.linalg.norm(a)


def l2_norm(a: np.array) -> float:
    """
    Calculate L2 norm for an array.

    Parameters
    ----------
        a (np.array): the array to calculate the L2 on

    Returns
    -------
        (float): the norm.
    """
    assert a.ndim == 1, "Check that 'l2_norm' receives a 1D array."
    return np.linalg.norm(a)


def linf_norm(a: np.array) -> float:
    """
    Calculate L-inf norm for an array.

    Parameters
    ----------
        a (np.array): the array to calculate the L-inf on.

    Returns
    -------
        (float): the norm.
    """
    assert a.ndim == 1, "Check that 'linf_norm' receives a 1D array."
    return np.linalg.norm(a, ord=np.inf)
