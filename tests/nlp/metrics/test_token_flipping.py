import numpy as np
import pytest

from quantus.nlp.metrics.faithfullness.token_flipping import TokenFlipping


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 39)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 39),
        ),
    ],
    ids=["pruning", "activation"],
)
def test_tf_model(
    tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    # assert not (result == np.NZERO).any()
    # assert not (result == np.PZERO).any()
    assert result.shape == expected_shape


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 42)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 42),
        ),
    ],
    ids=["pruning", "activation"],
)
def test_torch_model(
    torch_sst2_model, sst2_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(torch_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    # assert not (result == np.NZERO).any()
    # assert not (result == np.PZERO).any()
    assert result.shape == expected_shape
