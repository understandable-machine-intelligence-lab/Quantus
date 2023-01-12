import numpy as np


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1, optionally making them all positive."""
    num_dim = len(scores.shape)
    if num_dim > 2:
        raise ValueError()
    if num_dim == 2:
        return np.asarray([normalize_sum_to_1(i) for i in scores])
    if num_dim < 1:
        return scores
    scores = scores + np.finfo(np.float32).eps
    return scores / np.abs(scores).sum(-1)
