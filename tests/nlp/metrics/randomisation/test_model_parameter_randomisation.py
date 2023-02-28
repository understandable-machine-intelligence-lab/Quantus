import numpy as np
import pytest
from typing import Dict
from quantus.nlp import ModelParameterRandomisation
from tests.nlp.utils import skip_in_ci


@skip_in_ci
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
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["raw scores", "sample correlation"],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
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


@skip_in_ci
@pytest.mark.nlp
@pytest.mark.pytorch_model
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
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["raw scores", "sample correlation"],
)
def test_torch_emotion_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    if init_kwargs.get("return_sample_correlation"):
        assert isinstance(result, np.ndarray)
        result = [result]
    else:
        assert isinstance(result, Dict)
        result = result.values()

    for i in result:
        assert not (i == np.NINF).any()
        assert not (i == np.PINF).any()
        assert not (i == np.NAN).any()
        assert not (i == np.NZERO).any()
        assert not (i == np.PZERO).any()


@skip_in_ci
@pytest.mark.nlp
@pytest.mark.pytorch_model
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
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["raw scores", "sample correlation"],
)
def test_torch_fnet_model(torch_fnet, sst2_dataset, init_kwargs, call_kwargs):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(torch_fnet, sst2_dataset, **call_kwargs)
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
