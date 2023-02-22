from typing import List
import numpy as np
import pytest
from quantus.nlp.helpers.utils import batch_list, pad_ragged_vector


@pytest.mark.nlp
@pytest.mark.utils
def test_batch_list():
    flat_list = list(range(1000))
    batched_list = batch_list(flat_list, batch_size=32)
    assert isinstance(batched_list, List)
    for index, element in enumerate(batched_list):
        assert isinstance(element, List)
        if index != len(batched_list) - 1:
            assert len(element) == 32


@pytest.mark.nlp
@pytest.mark.utils
def test_list_is_divisible():
    flat_list = list(range(960))
    batched_list = batch_list(flat_list, batch_size=32)
    assert isinstance(batched_list, List)
    for index, element in enumerate(batched_list):
        assert isinstance(element, List)
        assert len(element) == 32


@pytest.mark.nlp
@pytest.mark.utils
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (5, 5),
        (5, 7),
        (7, 5),
        ((10, 5), (11, 5)),
        ((11, 5), (10, 5)),
        ((8, 4), (8, 5)),
        ((8, 5), (8, 4)),
        ((8, 39, 5), (8, 40, 5)),
        ((8, 40, 5), (8, 39, 5)),
        ((8, 7, 39, 5), (8, 7, 40, 5)),
        ((8, 7, 40, 5), (8, 7, 39, 5)),
    ],
)
def test_pad(a_shape, b_shape):
    a = np.random.uniform(size=a_shape)
    b = np.random.uniform(size=b_shape)
    a_padded, b_padded = pad_ragged_vector(a, b)
    assert a_padded.shape == b_padded.shape
