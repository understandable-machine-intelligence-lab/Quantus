import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from quantus.nlp import (
    MaxSensitivity,
    PerturbationType,
)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        # uniform noise
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"perturbation_type": PerturbationType.latent_space},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain_text", "latent_space"],
)
def test_max_sensitivity_huggingface_model_tf(model, x_batch, init_kwargs, call_kwargs):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            lazy_fixture("fnet_ag_news_model"),
            lazy_fixture("ag_news_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        # uniform noise
        (
            lazy_fixture("fnet_ag_news_model"),
            lazy_fixture("ag_news_dataset"),
            {"perturbation_type": PerturbationType.latent_space},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain_text", "latent_space"],
)
def test_max_sensitivity_keras_model(model, x_batch, init_kwargs, call_kwargs):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        # uniform noise
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"perturbation_type": PerturbationType.latent_space},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
    ids=["plain_text", "latent_space"],
)
def test_max_sensitivity_huggingface_model_torch(
    model, x_batch, init_kwargs, call_kwargs
):
    metric = MaxSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()
