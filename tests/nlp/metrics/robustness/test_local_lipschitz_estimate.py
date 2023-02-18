import numpy as np
import pytest
from quantus.nlp import LocalLipschitzEstimate, PerturbationType, uniform_noise


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        # uniform noise
        (
            {
                "perturbation_type": PerturbationType.latent_space,
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = LocalLipschitzEstimate(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert not (np.asarray(result) == np.NINF).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        # uniform noise
        (
            {
                "perturbation_type": PerturbationType.latent_space,
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_torch_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = LocalLipschitzEstimate(nr_samples=5, **init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert not (np.asarray(result) == np.NINF).all()
