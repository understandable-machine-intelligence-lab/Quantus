from typing import List

from quantus.nlp.helpers.utils import batch_list


def test_batch_list(sst2_dataset_huge_batch):
    flat_list = sst2_dataset_huge_batch[:1000]
    batched_list = batch_list(flat_list, batch_size=32)
    assert isinstance(batched_list, List)
    for index, element in enumerate(batched_list):
        assert isinstance(element, List)
        if index != len(batched_list) - 1:
            assert len(element) == 32


def test_list_is_divisible(sst2_dataset_huge_batch):
    flat_list = sst2_dataset_huge_batch
    batched_list = batch_list(flat_list, batch_size=32)
    assert isinstance(batched_list, List)
    for index, element in enumerate(batched_list):
        assert isinstance(element, List)
        assert len(element) == 32
