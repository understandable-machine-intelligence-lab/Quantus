import numpy as np
from pytest_lazyfixture import lazy_fixture
import pytest


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset_huge_batch"),
            lazy_fixture("tf_distilbert_sst2_model"),
        ),
        (
            lazy_fixture("sst2_dataset_huge_batch"),
            lazy_fixture("torch_distilbert_sst2_model"),
        ),
    ],
)
def test_predict_on_huge_batch(x_batch, model):
    logits = model.predict(x_batch)
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (len(x_batch), 2)
