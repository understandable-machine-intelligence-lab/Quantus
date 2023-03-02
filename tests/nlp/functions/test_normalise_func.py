import pytest
import numpy as np
import tensorflow as tf
from quantus.nlp import normalize_sum_to_1


@pytest.mark.nlp
@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "a_batch",
    [
        np.random.default_rng(0).normal(size=8),
        np.random.default_rng(0).normal(size=(8, 32)),
        tf.random.normal(shape=(8,)),
        tf.random.normal(shape=(8, 32)),
    ],
    ids=["1D", "2D", "1D-TF", "2D-TF"],
)
def test_normalise_func(a_batch):
    result = normalize_sum_to_1(a_batch)
    assert np.allclose(np.sum(np.abs(result), axis=-1), 1.0)


@pytest.mark.nlp
@pytest.mark.normalise_func
def test_invalid_input_shape():
    x_batch = np.random.default_rng(0).normal(size=(8, 32, 32))
    with pytest.raises(ValueError):
        normalize_sum_to_1(x_batch)
