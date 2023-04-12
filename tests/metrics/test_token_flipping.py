import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.explanation_func import explain
from quantus.metrics.faithfulness import TokenFlipping


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model,tokenizer,init_kwargs",
    [
        (
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_tokenizer"),
            {"normalise": True},
        ),
        (
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_tokenizer"),
            {"normalise": True, "task": "activation"},
        ),
    ],
    ids=["pruning_tf", "activation_torch"],
)
def test_token_flipping(model, tokenizer, sst2_dataset, init_kwargs):
    metric = TokenFlipping(**init_kwargs)
    result = metric(
        model, **sst2_dataset, a_batch=None, explain_func=explain, tokenizer=tokenizer
    )
    assert isinstance(result, np.ndarray)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    # assert not (result == np.NZERO).any()
    # assert not (result == np.PZERO).any()
    assert result.shape == (29,)


def test_2_batches():
    pytest.fail("Not Implemented")
