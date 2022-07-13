from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain
from ...quantus.helpers.pytorch_model import PyTorchModel
from ...quantus.helpers.tf_model import TensorFlowModel


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "normalise": True,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
                "return_aggregate": True,
            },
            1.0,
        ),
    ],
)
def test_completeness(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = Completeness(**params)(
        model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, **params
    )

    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "n_samples": 1,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
                "return_aggregate": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "n_samples": 2,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 2,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": True,
            },
            1.0,
        ),
    ],
)
def test_non_sensitivity(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = NonSensitivity(**params)(
        model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, **params
    )
    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Gradient",
                "input_shift": -1,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Gradient",
                "input_shift": -1,
                "disable_warnings": False,
                "display_progressbar": False,
                "features_in_step": 112,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputxGradient",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
                "return_aggregate": True,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputxGradient",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "features_in_step": 112,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "features_in_step": 112,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": True,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
                "features_in_step": 112,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputXGradient",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputXGradient",
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": True,
                "features_in_step": 112,
            },
            {"dtypes": [True, False]},
        ),
    ],
)
def test_input_invariance(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = InputInvariance(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    assert np.all([s in expected["dtypes"] for s in scores]), "Test failed."
