import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.explanation_func import explain
from quantus.metrics.faithfulness import TokenFlipping
from quantus.helpers.utils import get_wrapped_model


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, init_kwargs",
    [
        (lazy_fixture("tf_sst2_model"), {"normalise": True}),
        (
            lazy_fixture("torch_sst2_model"),
            {"normalise": True, "task": "activation"},
        ),
    ],
    ids=["pruning", "activation"],
)
def test_tf_model(model, sst2_dataset, init_kwargs):
    expected_shape = (
        get_wrapped_model(model)
        .tokenizer.batch_encode(sst2_dataset["x_batch"])["input_ids"][0]
        .shape
    )
    metric = TokenFlipping(**init_kwargs)
    result = metric(
        model, **sst2_dataset, a_batch=None, explain_func=explain
    )
    assert isinstance(result, np.ndarray)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    # assert not (result == np.NZERO).any()
    # assert not (result == np.PZERO).any()
    assert result.shape == expected_shape
