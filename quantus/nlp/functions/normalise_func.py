from scipy.special import softmax
import numpy as np


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    if scores.ndim > 2:
        raise ValueError("Only 2D and 1D inputs are supported.")
    if scores.ndim == 2:
        return np.asarray([normalize_sum_to_1(i) for i in scores])
    scores = scores + np.finfo(np.float32).eps
    return scores / np.abs(scores).sum(-1)
