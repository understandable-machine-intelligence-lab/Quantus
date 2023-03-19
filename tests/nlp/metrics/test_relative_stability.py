import numpy as np
import pytest

from quantus.nlp import (
    RelativeInputStability,
    RelativeOutputStability,
    RelativeRepresentationStability,
    typo_replacement,
    uniform_noise,
)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "perturb_func": typo_replacement},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
        (
            {
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_ris_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeInputStability(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "perturb_func": typo_replacement},
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
def test_ros_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeOutputStability(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "perturb_func": typo_replacement},
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
def test_rrs_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeRepresentationStability(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.slow
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {
                "explain_func_kwargs": {
                    "method": "SHAP",
                    "call_kwargs": {"max_evals": 5},
                }
            },
        ),
        (
            {
                "perturb_func": uniform_noise,
            },
            {"explain_func_kwargs": {"method": "GradXInput"}},
        ),
    ],
    ids=["plain text", "latent space"],
)
def test_ris_torch_model(torch_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeInputStability(nr_samples=5, **init_kwargs)
    result = metric(torch_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
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
def test_ros_torch_model(torch_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeOutputStability(nr_samples=5, **init_kwargs)
    result = metric(torch_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
@pytest.mark.slow
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {
                "explain_func_kwargs": {
                    "method": "IntGrad",
                }
            },
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
def test_rrs_torch_model(torch_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeRepresentationStability(nr_samples=5, **init_kwargs)
    result = metric(torch_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
