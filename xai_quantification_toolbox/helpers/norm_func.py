"""Collection of norm functions i..e, ways to measure the norm of a input- (or explanation) vector."""
import numpy as np


def fro_norm(a: np.array) -> float:
    """Calculate Froberius norm for an array."""
    assert a.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return np.linalg.norm(a)


def l2_norm(a: np.array) -> float:
    """Calculate L2-norm for an array."""
    assert a.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return np.linalg.norm(a, ord="2")


def l1_norm(a: np.array) -> float:
    """Calculate L1-norm for an array."""
    assert a.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return np.linalg.norm(a, ord="1")


def linf_norm(a: np.array) -> float:
    """Calculate L inf-norm for an array."""
    assert a.ndim == 1, "Check that 'perturb_func' receives a 1D array."
    return np.linalg.norm(a, ord=np.inf)


NORM_FUNCTIONS = {
    "fro_norm": fro_norm,
    "l2_norm": l2_norm,
    "l1_norm": l1_norm,
    "inf_norm": linf_norm
}
