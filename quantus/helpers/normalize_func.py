import numpy as np


def normalize_by_max(a: np.ndarray) -> np.ndarray:
    """"Normalize attributions by the maximum absolute value of the explanation."""
    a /= np.max(np.abs(a))
    return a
