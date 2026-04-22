from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
import torch
import torch.nn as nn

from quantus.functions.explanation_func import explain
from quantus.metrics.axiomatic import Completeness, InputInvariance, NonSensitivity

# test_axiomatic_metrics.py  (or similar)

def _ensure_4d(x):
    """Make sure x is (B, C, H, W), even if passed as (B, N)."""
    x = np.array(x)
    if x.ndim == 2:  
        B, N = x.shape
        side = int(np.sqrt(N))
        x = x.reshape(B, 1, side, side)
    elif x.ndim == 3:  
        x = x[:, None, :, :]
    return x

class SensitiveModel(nn.Module):
    def shape_input(self, x, shape, channel_first=True, batched=True):
        return x

    def forward(self, x):
        x = _ensure_4d(x)
        return x.sum(axis=(1, 2, 3), keepdims=True)

    def predict(self, x):
        x = _ensure_4d(x)
        return self.forward(x)

class InsensitiveModel(nn.Module):
    def shape_input(self, x, shape, channel_first=True, batched=True):
        return x
    def forward(self, x):
        x = _ensure_4d(x)
        B = x.shape[0]
        return np.ones((B, 1), dtype=float) * 100.0
    def predict(self, x):
        return self.forward(x)

class SemiSensitiveModel(nn.Module):
    def shape_input(self, x, shape, channel_first=True, batched=True):
        return x
    def forward(self, x):
        x = _ensure_4d(x)
        top_sum = x[:, :, 0, :].sum(axis=(1, 2))
        return top_sum[:, None]
    def predict(self, x):
        x = _ensure_4d(x)
        return self.forward(x)

class TrickModel(nn.Module):
    def shape_input(self, x, shape, channel_first=True, batched=True):
        return x
    def predict(self, x):
        x = _ensure_4d(x)
        bottom_sum = x[:, :, 1, :].sum(axis=(1, 2))
        return bottom_sum[:, None]

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
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "features_in_step": 2,
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
    "scenario,model_factory,x_batch,y_batch,a_batch,expected_violations,kwargs",
    [
        
        (
            "zero_violations",
            lambda: SemiSensitiveModel(),
            np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=float),
            np.array([0]),
            np.array([[[[10.0, 10.0], [0.0, 0.0]]]], dtype=float),
            0,
            {"features_in_step": 2, "eps": 1e-5},
        ),
        (
            "low_attr_high_change",
            lambda: SensitiveModel(),
            np.array([[[[5.0, 5.0], [5.0, 5.0]]]], dtype=float),
            np.array([0]),
            np.random.uniform(1e-6, 2e-6, size=(1, 1, 2, 2)),
            4,
            {"features_in_step": 2, "eps": 1e-5},
        ),
        (
            "high_attr_low_change",
            lambda: InsensitiveModel(),
            np.random.rand(1, 1, 4, 4),
            np.array([0]),
            np.ones((1, 1, 4, 4)),
            16,
            {"features_in_step": 2, "eps": 1e-5},
        ),
        (
            "half_good_half_bad",
            lambda: TrickModel(),
            np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=float),
            np.array([0]),
            np.array([[[[10.0, 10.0], [0.0, 0.0]]]], dtype=float),
            4,
            {"features_in_step": 1, "eps": 1e-5},
        ),
    ],
)
def test_my_non_sensitivity_logics(
    scenario,
    model_factory,
    x_batch,
    y_batch,
    a_batch,
    expected_violations,
    kwargs,
):
    """
    Parametrized logic-based tests for NonSensitivity.
    Each scenario defines a different consistency pattern between attribution and model behavior.
    """
    model = model_factory()
    model.eval()
    metric = NonSensitivity(
        disable_warnings=True,
        perturb_baseline="uniform",
        normalise=False,
        **kwargs,
    )

    scores = metric.evaluate_batch(model, x_batch, y_batch, a_batch)

    # --- Assertions ---
    assert isinstance(scores, np.ndarray), f"[{scenario}] Output must be np.ndarray"
    assert scores.shape[0] == x_batch.shape[0], f"[{scenario}] Wrong batch size"
    assert np.all(np.isfinite(scores)), f"[{scenario}] Scores contain NaN/Inf"

    if expected_violations is not None:
        assert scores[0] == expected_violations, (
            f"[{scenario}] expected {expected_violations}, got {scores[0]}"
        )

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
                        "method": "InputXGradient",
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
