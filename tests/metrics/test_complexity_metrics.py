from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain


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
        a_batch=data["a_batch"],
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
        a_batch=data["a_batch"],
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
        a_batch=data["a_batch"],
        **call_params
    )
    assert scores is not None, "Test failed."
