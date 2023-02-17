import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from quantus.nlp import TokenFlipping, PerturbationType, uniform_noise


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_huggingface_model_tf(model, x_batch, init_kwargs, call_kwargs):
    metric = TokenFlipping(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        (
            lazy_fixture("tf_distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"abs": True, "normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_huggingface_model_torch(model, x_batch, init_kwargs, call_kwargs):
    metric = TokenFlipping(nr_samples=5, **init_kwargs)
    result = metric(model, x_batch, **call_kwargs)  # noqa
    assert not (np.asarray(result) == 0).all()
