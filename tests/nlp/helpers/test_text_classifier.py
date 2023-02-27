import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.parametrize(
    "x_batch", [lazy_fixture("sst2_dataset"), np.random.uniform(size=(8, 30, 768))]
)
def test_get_hidden_representations_tf(tf_sst2_model, x_batch):
    result = tf_sst2_model.get_hidden_representations(x_batch)
    assert isinstance(result, np.ndarray)


@pytest.mark.nlp
@pytest.mark.pytorch_model
def test_get_hidden_representations_torch(emotion_model, emotion_dataset):
    result = emotion_model.get_hidden_representations(emotion_dataset)
    assert isinstance(result, np.ndarray)
