import numpy as np
import pytest
from quantus.nlp import MaxSensitivity, PerturbationType, uniform_noise
from tests.nlp.util import skip_on_apple_silicon


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            {"abs": True, "normalise": True},
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
    ids=["plain_text", "latent_space"],
)
def test_tf_model(tf_distilbert_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(tf_distilbert_sst2_model, sst2_dataset, **call_kwargs)
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            {"abs": True, "normalise": True},
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
    ids=["plain_text", "latent_space"],
)
def test_torch_model(
    torch_distilbert_sst2_model, sst2_dataset, init_kwargs, call_kwargs
):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(torch_distilbert_sst2_model, sst2_dataset, **call_kwargs)
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@skip_on_apple_silicon
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            {"abs": True, "normalise": True},
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
    ids=["plain_text", "latent_space"],
)
def test_keras_model(fnet_ag_news_model, ag_news_dataset, init_kwargs, call_kwargs):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(fnet_ag_news_model, ag_news_dataset, **call_kwargs)
    assert not (np.asarray(result) == 0).all()
