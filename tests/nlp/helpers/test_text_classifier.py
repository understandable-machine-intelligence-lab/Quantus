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
    assert isinstance(result, tf.Tensor)
    assert len(result.shape) == 4


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch",
    [lazy_fixture("sst2_dataset"), torch.tensor(np.random.uniform(size=(8, 30, 768)))],
)
def test_get_hidden_representations_torch(emotion_model, x_batch):
    result = emotion_model.get_hidden_representations(x_batch)
    assert isinstance(result, torch.Tensor)
