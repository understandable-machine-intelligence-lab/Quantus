import numpy as np
import pytest

from quantus.functions.explanation_func import explain
from quantus.metrics.faithfulness import TokenFlipping


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (39,)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (39,),
        ),
    ],
    ids=["pruning", "activation"],
)
def test_tf_model(
    tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(
        tf_sst2_model, **sst2_dataset, **call_kwargs, a_batch=None, explain_func=explain
    )
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
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (39,)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (39,),
        ),
    ],
    ids=["pruning", "activation"],
)
def test_torch_model(
    torch_sst2_model,
    sst2_dataset,
    init_kwargs,
    call_kwargs,
    expected_shape,
    torch_device,
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(
        torch_sst2_model,
        **sst2_dataset,
        **call_kwargs,
        device=torch_device,
        a_batch=None,
        explain_func=explain
    )
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    # assert not (result == np.NZERO).any()
    # assert not (result == np.PZERO).any()
    assert result.shape == expected_shape
