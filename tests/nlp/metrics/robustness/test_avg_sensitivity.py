import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from quantus.nlp import (
    AvgSensitivity,
    PerturbationType,
    synonym_replacement,
    typo_replacement,
    gaussian_noise,
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
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # synonym_replacement
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # typo_replacement
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # uniform noise
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"noise_type": "latent"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        #(
        #    lazy_fixture("tf_distilbert_sst2_model"),
        #    lazy_fixture("sst2_dataset"),
        #    {"noise_type": "latent"},
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("tf_distilbert_sst2_model"),
        #    lazy_fixture("sst2_dataset"),
        #    {"noise_type": "latent"},
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
    ],
    ids=[
        #"spelling_replacement, GradNorm",
        #"spelling_replacement, InputXGrad",
        #"spelling_replacement, IntGrad",
        #"synonym_replacement, GradNorm",
        #"synonym_replacement, InputXGrad",
        #"synonym_replacement, IntGrad",
        #"typo_replacement, GradNorm",
        #"typo_replacement, InputXGrad",
        #"typo_replacement, IntGrad",
        #"uniform noise, GradNorm",
        #"uniform noise, InputXGrad",
        #"uniform noise, IntGrad",
    ],
)
def test_average_sensitivity_huggingface_model_tf(
    model, x_batch, init_kwargs, call_kwargs
):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        # spelling_replacement
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"abs": True, "normalise": True},
        #    {"explain_func_kwargs": {"method": "GradNorm"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True},
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True},
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
        # synonym_replacement
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": synonym_replacement},
        #    {"explain_func_kwargs": {"method": "GradNorm"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": synonym_replacement},
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": synonym_replacement},
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
        # typo_replacement
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": typo_replacement},
        #    {"explain_func_kwargs": {"method": "GradNorm"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": typo_replacement},
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"normalise": True, "perturb_func": typo_replacement},
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
        # uniform noise
        (
            lazy_fixture("fnet_ag_news_model"),
            lazy_fixture("ag_news_dataset"),
            {"normalise": True, "perturbation_type": PerturbationType.latent_space},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"perturbation_type": PerturbationType.latent_space},
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {"perturbation_type": PerturbationType.latent_space},
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
        # gaussian noise
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {
        #        "perturbation_type": PerturbationType.latent_space,
        #        "perturb_func": gaussian_noise,
        #    },
        #    {"explain_func_kwargs": {"method": "GradNorm"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {
        #        "perturbation_type": PerturbationType.latent_space,
        #        "perturb_func": gaussian_noise,
        #    },
        #    {"explain_func_kwargs": {"method": "InputXGrad"}},
        #),
        #(
        #    lazy_fixture("fnet_ag_news_model"),
        #    lazy_fixture("ag_news_dataset"),
        #    {
        #        "perturbation_type": PerturbationType.latent_space,
        #        "perturb_func": gaussian_noise,
        #    },
        #    {"explain_func_kwargs": {"method": "IntGrad"}},
        #),
    ],
    ids=[
        #"spelling_replacement, GradNorm",
        #"spelling_replacement, InputXGrad",
        #"spelling_replacement, IntGrad",
        #"synonym_replacement, GradNorm",
        #"synonym_replacement, InputXGrad",
        #"synonym_replacement, IntGrad",
        #"typo_replacement, GradNorm",
        #"typo_replacement, InputXGrad",
        #"typo_replacement, IntGrad",
        #"uniform noise, GradNorm",
        #"uniform noise, InputXGrad",
        #"uniform noise, IntGrad",
        #"gaussian noise, GradNorm",
        #"gaussian noise, InputXGrad",
        #"gaussian noise, IntGrad",
    ],
)
def test_average_sensitivity_keras_model(model, x_batch, init_kwargs, call_kwargs):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        # spelling_replacement
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # synonym_replacement
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "synonym_replacement"},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # typo_replacement
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"normalise": True, "perturb_func": "typo_replacement"},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        # uniform noise
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"noise_type": "latent"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"noise_type": "latent"},
            {"explain_func_kwargs": {"method": "InputXGrad"}},
        ),
        (
            lazy_fixture("torch_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"noise_type": "latent"},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
    ],
    ids=[
        "spelling_replacement, GradNorm",
        "spelling_replacement, InputXGrad",
        "spelling_replacement, IntGrad",
        "synonym_replacement, GradNorm",
        "synonym_replacement, InputXGrad",
        "synonym_replacement, IntGrad",
        "typo_replacement, GradNorm",
        "typo_replacement, InputXGrad",
        "typo_replacement, IntGrad",
        "uniform noise, GradNorm",
        "uniform noise, InputXGrad",
        "uniform noise, IntGrad",
    ],
)
def test_average_sensitivity_huggingface_model_torch(
    model, x_batch, init_kwargs, call_kwargs
):
    metric = AvgSensitivity(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()
