from typing import List
import numpy as np
import pytest
from quantus.nlp.helpers.utils import (
    batch_list,
    get_logits_for_labels,
)


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


logits = np.random.uniform(size=(8, 2))
y_batch = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
expected_logits = np.asarray([i[j] for i, j in zip(logits, y_batch)])


@pytest.mark.utils
def test_get_logits_for_labels():
    result = get_logits_for_labels(logits, y_batch)
    assert (result == expected_logits).all()
