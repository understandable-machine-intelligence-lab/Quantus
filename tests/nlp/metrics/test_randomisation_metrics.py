import numpy as np
import pytest
from typing import Dict
from quantus.nlp import RandomLogit, ModelParameterRandomisation


@pytest.mark.nlp
@pytest.mark.pytorch_model
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "num_classes": 2},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_random_logit_torch_model(
    torch_sst2_model, sst2_dataset, init_kwargs, call_kwargs
):
    metric = RandomLogit(**init_kwargs)
    result = metric(torch_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "num_classes": 2},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
    ],
)
def test_random_logit_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RandomLogit(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            {"normalise": True, "return_sample_correlation": True},
            {"explain_func_kwargs": {"method": "GradXInput"}},
        ),
    ],
    ids=["raw scores", "sample correlation"],
)
def test_model_parameter_randomisation_tf_model(
    tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs
):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    if not init_kwargs.get("return_sample_correlation"):
        assert isinstance(result, Dict)
        for i in result.values():
            assert isinstance(i, np.ndarray)
            assert not (i == np.NINF).any()
            assert not (i == np.PINF).any()
            assert not (i == np.NAN).any()
            assert not (i == np.NZERO).any()
            assert not (i == np.PZERO).any()
    else:
        assert isinstance(result, np.ndarray)
        assert not (result == np.NINF).any()
        assert not (result == np.PINF).any()
        assert not (result == np.NAN).any()
        assert not (result == np.NZERO).any()
        assert not (result == np.PZERO).any()
