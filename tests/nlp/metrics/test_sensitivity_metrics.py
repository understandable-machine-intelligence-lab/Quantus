import pytest
from pytest_lazyfixture import lazy_fixture
from quantus.nlp.metrics.robustness.avg_sensitivity import AvgSensitivity
from quantus.nlp.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.nlp.functions.explanation_function import explain


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        (
            lazy_fixture("distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {},
            {"explain_func": explain, "explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_average_sensitivity_huggingface_model(
    model, x_batch, init_kwargs, call_kwargs, capsys
):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        metric = AvgSensitivity(nr_samples=5, **init_kwargs)
        result = metric.__call__(model, x_batch, y_batch, **call_kwargs)
        print(f"{result = }")


@pytest.mark.nlp
@pytest.mark.parametrize(
    "model, x_batch, init_kwargs, call_kwargs",
    [
        (
            lazy_fixture("distilbert_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {},
            {"explain_func": explain, "explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_max_sensitivity_huggingface_model(
    model, x_batch, init_kwargs, call_kwargs, capsys
):
    with capsys.disabled():
        y_batch = model.predict(x_batch).argmax(axis=-1)
        metric = MaxSensitivity(nr_samples=5, **init_kwargs)
        result = metric.__call__(model, x_batch, y_batch, **call_kwargs)
        print(f"{result = }")
