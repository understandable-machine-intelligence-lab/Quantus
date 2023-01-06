import numpy as np

DEFAULT_KERNEL_WIDTH = 25


def exponential_kernel(
    distance: float, kernel_width: float = DEFAULT_KERNEL_WIDTH
) -> np.ndarray:
    """The exponential kernel."""
    return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def normalize_scores(scores: np.ndarray, make_positive: bool = False) -> np.ndarray:
    """Makes the absolute values sum to 1, optionally making them all positive."""
    if len(scores.shape) == 2:
        return np.asarray([normalize_scores(i) for i in scores])
    if len(scores) < 1:
        return scores
    scores = scores + np.finfo(np.float32).eps
    if make_positive:
        scores = np.abs(scores)
    return scores / np.abs(scores).sum(-1)
