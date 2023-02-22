from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np


from quantus.functions.explanation_func import explain
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.complexity import Complexity, EffectiveComplexity, Sparseness


def explain_func_stub(*args, **kwargs):
    # tf-explain does not support 2D inputs
    input_shape = kwargs.get("inputs").shape
    return np.random.uniform(low=0, high=0.5, size=input_shape)


@pytest.mark.complexity
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "reduce_axes": (),
                    },
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_tf"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {"explain_func": explain_func_stub},
            },
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_sparseness(
    model: ModelInterface,
    data: dict,
    params: dict,
    expected: Union[float, dict, bool],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = Sparseness(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data.get("a_batch"),
        **call_params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.complexity
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "reduce_axes": (),
                    },
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_tf"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {"explain_func": explain_func_stub},
            },
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_complexity(
    model: ModelInterface,
    data: dict,
    params: dict,
    expected: Union[float, dict, bool],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = Complexity(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data.get("a_batch"),
        **call_params
    )
    assert scores is not None, "Test failed."


@pytest.mark.complexity
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_1d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            None,
            lazy_fixture("almost_uniform_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "reduce_axes": (),
                    },
                },
            },
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("titanic_model_tf"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {"explain_func": explain_func_stub},
            },
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_effective_complexity(
    model: ModelInterface,
    data: dict,
    params: dict,
    expected: Union[float, dict, bool],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = EffectiveComplexity(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data.get("a_batch"),
        **call_params
    )
    assert scores is not None, "Test failed."
