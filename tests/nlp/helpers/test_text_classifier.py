import numpy as np
import pytest


@pytest.mark.nlp
@pytest.mark.tf_model
def test_get_hidden_representations_tf(tf_sst2_model, sst2_dataset):
    result = tf_sst2_model.get_hidden_representations(sst2_dataset)
    assert isinstance(result, np.ndarray)


@pytest.mark.nlp
@pytest.mark.pytorch_model
def test_get_hidden_representations_torch(emotion_model, emotion_dataset):
    result = emotion_model.get_hidden_representations(emotion_dataset)
    assert isinstance(result, np.ndarray)