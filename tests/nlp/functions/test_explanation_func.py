import numpy as np
import tensorflow as tf
import pytest
from pytest_lazyfixture import lazy_fixture
from typing import List
from quantus.nlp import explain, NoiseType


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
        ),
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
        ),
    ],
)
def test_explain_grad_norm(x_batch, model):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = explain(model, x_batch, y_batch)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
        ),
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
        ),
    ],
)
def test_explain_input_x_gradient(x_batch, model):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = explain(model, x_batch, y_batch, method="GradXInput")
    assert isinstance(a_batch, List)
    assert len(a_batch) == len(y_batch)


def unknown_token_baseline_function(x: tf.Tensor) -> np.ndarray:
    return np.load("tests/assets/nlp/unknown_token_embedding.npy")


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
            {},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
            {"baseline_fn": unknown_token_baseline_function},
        ),
        (lazy_fixture("ag_news_dataset"), lazy_fixture("fnet_ag_news_model"), {}),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
            {},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
            {"baseline_fn": unknown_token_baseline_function},
        ),
    ],
)
def test_explain_integrated_gradients(x_batch, model, kwargs):
    y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
    a_batch = explain(model, x_batch, y_batch, method="IntGrad", **kwargs)  # noqa
    assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
            {"explain_fn": "GradXInput"},
        ),
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
            {"noise_type": NoiseType.additive},
        ),
    ],
)
def test_explain_noise_grad(x_batch, model, kwargs):
    y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
    a_batch = explain(
        model, x_batch, y_batch, method="NoiseGrad++", n=2, m=2, **kwargs  # noqa
    )
    assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
            {"explain_fn": "GradNorm"},
        ),
    ],
)
def test_explain_noise_grad_torch(x_batch, model, kwargs):
    y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
    a_batch = explain(
        model,
        x_batch,
        y_batch,
        method="NoiseGrad++",
        init_kwargs={"n": 2, "m": 2},
        **kwargs  # noqa
    )
    assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
        ),
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
        ),
    ],
)
def test_explain_lime(x_batch, model):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = explain(
        model, x_batch, y_batch, method="LIME", call_kwargs={"num_samples": 5}
    )
    assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("tf_distilbert_sst2_model"),
        ),
        (
            lazy_fixture("ag_news_dataset"),
            lazy_fixture("fnet_ag_news_model"),
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("torch_distilbert_sst2_model"),
        ),
    ],
)
def test_explain_shap(x_batch, model):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = explain(
        model, x_batch, y_batch, method="SHAP", call_kwargs={"max_evals": 5}
    )
    assert len(a_batch) == len(y_batch)
