from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.metrics.axiomatic import Completeness, InputInvariance, NonSensitivity


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "return_aggregate": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
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

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    scores = Completeness(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
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
                "a_batch_generate": False,
                "init": {
                    "n_samples": 1,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "n_samples": 1,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "n_samples": 1,
                    "eps": 1e-2,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "n_samples": 1,
                    "eps": 1e-2,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "n_samples": 2,
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "n_samples": 2,
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "n_samples": 1,
                    "eps": 1e-10,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "n_samples": 1,
                    "eps": 1e-10,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
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

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    scores = NonSensitivity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "a_batch_generate": True,
                    "init": {
                        "abs": False,
                        "normalise": False,
                        "input_shift": 0.2,
                        "disable_warnings": True,
                        "display_progressbar": False,
                    },
                    "call": {
                        "explain_func": explain,
                        "explain_func_kwargs": {
                            "method": "VanillaGradients",
                        },
                    },
                },
                {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -0.9,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "InputXGradient",},
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": True,
                    "normalise": True,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "InputXGradient",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {"method": "InputXGradient",},
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": True,
                    "normalise": True,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "abs": False,
                    "normalise": False,
                    "input_shift": -1,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "InputXGradient",
                    },
                },
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

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    scores = InputInvariance(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert np.all([s in expected["dtypes"] for s in scores]), "Test failed."
