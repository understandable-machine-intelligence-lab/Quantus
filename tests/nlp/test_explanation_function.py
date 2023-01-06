import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.nlp.explanation_function import (
    tf_explain_grad_norm,
    tf_explain_int_grad,
    tf_explain_lime,
    tf_explain_shap,
    tf_explain_input_x_gradient,
    tf_explain_noise_grad,
)


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
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_grad_norm(x_batch, y_batch, model)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")


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
def test_explain_integrated_gradients(x_batch, model, capsys):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_int_grad(x_batch, y_batch, model)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")


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
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_lime(x_batch, y_batch, model, num_samples=10)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")


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
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_shap(x_batch, y_batch, model)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")


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
def test_explain_noise_grad(x_batch, model, capsys):
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_noise_grad(x_batch, y_batch, model)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")


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
    y_batch = model.predict(x_batch).argmax(axis=-1)
    a_batch = tf_explain_input_x_gradient(x_batch, y_batch, model)

    assert len(a_batch) == len(y_batch)
    with capsys.disabled():
        print(f"{a_batch = }")
