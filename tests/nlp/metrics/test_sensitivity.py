import numpy as np
import pytest

from quantus.nlp import AvgSensitivity, MaxSensitivity, uniform_noise, gaussian_noise
from tests.nlp.utils import skip_on_apple_silicon


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.robustness
@pytest.mark.parametrize(
    "init_kwargs",
    [
        {"abs": True},
        {"perturb_func": gaussian_noise},
    ],
    ids=["plain text", "latent space"],
)
def test_avg_sensitivity_tf(tf_sst2_model, sst2_dataset, init_kwargs):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@skip_on_apple_silicon
@pytest.mark.nlp
@pytest.mark.keras_nlp_model
@pytest.mark.robustness
@pytest.mark.parametrize(
    "init_kwargs",
    [
        {},
        {
            "abs": True,
            "perturb_func": uniform_noise,
        },
    ],
    ids=["plain text", "latent space"],
)
def test_avg_sensitivity_keras_model(tf_sst2_model, sst2_dataset, init_kwargs):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.pytorch_model
@pytest.mark.robustness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            {
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_max_sensitivity_torch_fnet(torch_fnet, sst2_dataset, init_kwargs, call_kwargs):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(torch_fnet, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.pytorch_model
@pytest.mark.robustness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            {
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_max_sensitivity_torch_emotion_model(
    emotion_model, emotion_dataset, init_kwargs, call_kwargs
):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()