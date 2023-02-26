import pytest
import numpy as np
from quantus.nlp import normalize_sum_to_1


@pytest.mark.nlp
@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "size",
    [
        (8,),
        (8, 32),
    ],
    ids=["1D", "2D"],
)
def test_normalise_func(size):
    x_batch = np.random.default_rng(0).normal(size=size)
    result = normalize_sum_to_1(x_batch)
    assert np.allclose(np.sum(np.abs(result), axis=-1), 1.0)


@pytest.mark.nlp
@pytest.mark.normalise_func
def test_invalid_input_shape():
    x_batch = np.random.default_rng(0).normal(size=(8, 32, 32))
    with pytest.raises(ValueError):
        normalize_sum_to_1(x_batch)
