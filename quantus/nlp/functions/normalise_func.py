from importlib import util
from functools import singledispatch
import numpy as np


@singledispatch
def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    if scores.ndim > 2:
        raise ValueError("Only 2D and 1D inputs are supported.")
    scores = scores + np.finfo(np.float32).eps
    return (scores.T / np.abs(scores).sum(axis=-1)).T


if util.find_spec("tensorflow"):
    import tensorflow as tf
    from quantus.nlp.helpers.utils import tf_function

    @normalize_sum_to_1.register(tf.Tensor)
    @tf_function
    def _(scores: tf.Tensor) -> tf.Tensor:
        scores = scores + tf.keras.backend.epsilon()
        return tf.transpose(
            tf.transpose(scores) / tf.reduce_sum(tf.abs(scores), axis=-1),
        )
