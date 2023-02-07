import numpy as np


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    num_dim = len(scores.shape)
    if num_dim > 2:
        raise ValueError("Only 2D and 1D inputs are supported.")
    if num_dim == 2:
        return np.asarray([normalize_sum_to_1(i) for i in scores])
    scores = scores + np.finfo(np.float32).eps
    return scores / np.abs(scores).sum(-1)
