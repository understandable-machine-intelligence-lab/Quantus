import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch",
    [
        (lazy_fixture("tf_sst2_model"), lazy_fixture("sst2_dataset")),
        (lazy_fixture("emotion_model"), lazy_fixture("emotion_dataset")),
    ],
)
def test_get_hidden_representations(model, x_batch):
    result = model.get_hidden_representations(x_batch)
    assert isinstance(result, np.ndarray)
