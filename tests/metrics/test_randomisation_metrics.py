from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain
from ...quantus.helpers.model_interface import ModelInterface


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
                "return_aggregate": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
                "return_aggregate": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_model_parameter_randomisation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    explain = params["explain_func"]
    if params.get("a_batch_generate", True):
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores_layers = ModelParameterRandomisation(**params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **params,
            )
        return

    scores_layers = ModelParameterRandomisation(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    if isinstance(expected, float):
        assert all(
            s == expected for layer, scores in scores_layers.items() for s in scores
        ), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"]))
            for layer, scores in scores_layers.items()
            for s in scores
        ), "Test failed."


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
                "return_aggregate": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "num_classes": 10,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
                "return_aggregate": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_random_logit(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    explain = params["explain_func"]
    if params.get("a_batch_generate", True):
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    else:
        a_batch = None
    scores = RandomLogit(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."
