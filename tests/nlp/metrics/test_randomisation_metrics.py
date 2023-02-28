import numpy as np
import pytest
from typing import Dict
from tests.nlp.utils import skip_in_ci, skip_on_apple_silicon
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
def test_random_logit_torch_fnet_model(
    torch_fnet, sst2_dataset, init_kwargs, call_kwargs
):
    metric = RandomLogit(**init_kwargs)
    result = metric(torch_fnet, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@skip_on_apple_silicon
@pytest.mark.nlp
@pytest.mark.keras_nlp_model
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
def test_random_logit_keras_fnet_model(
    fnet_keras, ag_news_dataset, init_kwargs, call_kwargs
):
    metric = RandomLogit(**init_kwargs)
    result = metric(fnet_keras, ag_news_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@skip_in_ci
@pytest.mark.nlp
@pytest.mark.tf_model
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
def test_random_logit_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RandomLogit(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@skip_on_apple_silicon
@pytest.mark.nlp
@pytest.mark.keras_nlp_model
@pytest.mark.randomisation
def test_model_parameter_randomisation_keras_fnet_model(fnet_keras, ag_news_dataset):
    metric = ModelParameterRandomisation()
    result = metric(fnet_keras, ag_news_dataset)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


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
def test_model_parameter_randomisation_torch_emotion_model(
    emotion_model, emotion_dataset, init_kwargs, call_kwargs
):
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
