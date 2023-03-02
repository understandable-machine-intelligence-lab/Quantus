import numpy as np
import tensorflow as tf
import torch
import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.parametrize(
    "x_batch", [lazy_fixture("sst2_dataset"), tf.random.uniform(shape=(8, 30, 768))]
)
def test_get_hidden_representations_tf(tf_sst2_model, x_batch):
    result = tf_sst2_model.get_hidden_representations(x_batch)
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 4


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch",
    [
        lazy_fixture("sst2_dataset"),
        torch.tensor(np.random.uniform(size=(8, 30, 768)), dtype=torch.float32),
    ],
)
def test_get_hidden_representations_torch(torch_sst2_model, x_batch):
    result = torch_sst2_model.get_hidden_representations(x_batch)
    assert isinstance(result, np.ndarray)
