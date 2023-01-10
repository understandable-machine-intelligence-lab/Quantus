import numpy as np
import tensorflow as tf
import pytest
from pytest_lazyfixture import lazy_fixture

from nlp.functions.tf_explanation_function import (
    tf_explain_gradient_norm,
    tf_explain_integrated_gradients,
    tf_explain_lime,
    tf_explain_shap,
    tf_explain_input_x_gradient,
    tf_explain_noise_grad_plus_plus,
    tf_explain_gradient_norm_over_embeddings,
    tf_explain_integrated_gradients_over_embeddings,
    tf_explain_attention,
)
from nlp.functions.explanation_function import explain


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
        )
    ],
)
def test_explain_grad_norm(x_batch, model, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        a_batch = tf_explain_gradient_norm(x_batch, y_batch, model)
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
        )
    ],
)
def test_explain_input_x_gradient(x_batch, model, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        a_batch = tf_explain_input_x_gradient(x_batch, y_batch, model)
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


def unknown_token_baseline_function(x: tf.Tensor) -> tf.Tensor:
    return tf.convert_to_tensor(np.load("tests/assets/nlp/unknown_token_embedding.npy"))


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (lazy_fixture("sst2_dataset"), lazy_fixture("distilbert_sst2_model"), {}),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"baseline_fn": unknown_token_baseline_function},
        ),
    ],
)
def test_explain_integrated_gradients(x_batch, model, kwargs, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
        a_batch = tf_explain_integrated_gradients(
            x_batch, y_batch, model, **kwargs  # noqa
        )
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"explain_fn": tf_explain_gradient_norm_over_embeddings},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {
                "explain_fn": tf_explain_integrated_gradients_over_embeddings,
                "noise_type": "additive",
            },
        ),
    ],
)
def test_explain_noise_grad(x_batch, model, kwargs, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
        a_batch = tf_explain_noise_grad_plus_plus(
            x_batch, y_batch, model, n=2, m=2, **kwargs  # noqa
        )
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
        )
    ],
)
def test_explain_lime(x_batch, model, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        a_batch = tf_explain_lime(x_batch, y_batch, model, num_samples=10)
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
        )
    ],
)
def test_explain_shap(x_batch, model, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        a_batch = tf_explain_shap(x_batch, y_batch, model, call_kwargs={"max_evals": 5})
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, explain_fn_kwargs",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "GradNorm"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "InputXGrad"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "IntGrad"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "NoiseGrad++", "m": 2, "n": 2},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "LIME", "num_samples": 5},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
        ),
    ],
)
def test_wrapper(x_batch, model, explain_fn_kwargs, capsys):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)  # noqa
        a_batch = explain(x_batch, y_batch, model, explain_fn_kwargs)  # noqa
        print(f"{a_batch = }")
        assert len(a_batch) == len(y_batch)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "x_batch, model, kwargs",
    [
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"attention_layer_index": "mean"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"attention_head_index": "mean"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"attention_from_token_index": "mean"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"attention_from_token_index": None, "attention_to_token_index": "mean"},
        ),
        (
            lazy_fixture("sst2_dataset"),
            lazy_fixture("distilbert_sst2_model"),
            {"attention_from_token_index": None, "attention_to_token_index": -1},
        ),
    ],
)
def test_attention_explanation(x_batch, model, kwargs):
    a_batch = tf_explain_attention(x_batch, None, model, **kwargs)  # noqa
    assert len(a_batch) == len(x_batch)
