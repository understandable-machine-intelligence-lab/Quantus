from scipy.special import softmax
import numpy as np


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    num_dim = len(scores.shape)
    scores = scores + np.finfo(np.float32).eps
    if num_dim == 1:
        return softmax(scores)
    if num_dim == 2:
        return softmax(scores, axis=-1)
    raise ValueError("Only 2D and 1D inputs are supported.")
