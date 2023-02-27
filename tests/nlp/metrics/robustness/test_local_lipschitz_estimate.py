import numpy as np
import pytest
from quantus.nlp import LocalLipschitzEstimate, PerturbationType, uniform_noise


@pytest.mark.nlp
@pytest.mark.tf_model
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
                "perturbation_type": PerturbationType.latent_space,
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = LocalLipschitzEstimate(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    assert result.shape == (8,)


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
                "perturbation_type": PerturbationType.latent_space,
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_torch_emotion_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = LocalLipschitzEstimate(nr_samples=5, **init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    assert result.shape == (8,)
