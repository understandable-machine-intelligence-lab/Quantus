import numpy as np
import pytest
from tests.nlp.utils import skip_on_apple_silicon
from quantus.nlp import RelativeInputStability, uniform_noise


@pytest.mark.nlp
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
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RelativeInputStability(nr_samples=5, **init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@skip_on_apple_silicon
@pytest.mark.nlp
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
def test_keras_model(fnet_keras, ag_news_dataset, init_kwargs, call_kwargs):
    metric = RelativeInputStability(nr_samples=5, **init_kwargs)
    result = metric(fnet_keras, ag_news_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()


@pytest.mark.nlp
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
def test_torch_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = RelativeInputStability(nr_samples=5, **init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
