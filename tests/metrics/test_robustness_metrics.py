from typing import Union, Dict
from pytest_lazyfixture import lazy_fixture
import pytest
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.functions.discretise_func import floating_points, rank, sign, top_n_sign
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.robustness import (
    AvgSensitivity,
    Consistency,
    Continuity,
    LocalLipschitzEstimate,
    MaxSensitivity,
)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "abs": True,
                    "normalise": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "VanillaGradients",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "abs": True,
                    "normalise": True,
                    "return_aggregate": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "VanillaGradients",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_max_sensitivity(
    model: ModelInterface,
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
    scores = MaxSensitivity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(((s >= expected["min"]) & (s <= expected["max"])) for s in scores), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_local_lipschitz_estimate(
    model: ModelInterface,
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
    scores = LocalLipschitzEstimate(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
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
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
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
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
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
            {"exception": ValueError},
        ),
    ],
)
def test_continuity(
    model: ModelInterface,
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

    if isinstance(expected, Dict):
        if "exception" in expected:
            with pytest.raises(expected["exception"]):
                Continuity(**init_params)(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    **call_params,
                )
            return

    scores = Continuity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "lower_bound": 0.2,
                    "nr_samples": 10,
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
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_avg_sensitivity(
    model: ModelInterface,
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
    scores = AvgSensitivity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(((s >= expected["min"]) & (s <= expected["max"])) for s in scores), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": floating_points,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": sign,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": top_n_sign,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": rank,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": rank,
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "return_aggregate": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_consistency(
    model: ModelInterface,
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
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **call_params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = Consistency(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )
        return

    scores = Consistency(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )[0]
    assert (scores >= expected["min"]) & (scores <= expected["max"]), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric,model,data,params",
    [
        (
            LocalLipschitzEstimate,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_std": 255,
                "perturb_mean": 255,
            },
        ),
        (
            LocalLipschitzEstimate,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "perturb_std": 255,
                "perturb_mean": 255,
            },
        ),
        (
            AvgSensitivity,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "lower_bound": 1.0,
                "upper_bound": 255.0,
            },
        ),
        (
            AvgSensitivity,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "lower_bound": 1.0,
                "upper_bound": 255.0,
            },
        ),
        (
            MaxSensitivity,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "lower_bound": 1.0,
                "upper_bound": 255.0,
            },
        ),
        (
            MaxSensitivity,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "lower_bound": 1.0,
                "upper_bound": 255.0,
            },
        ),
    ],
)
def test_return_nan_when_prediction_changes(metric, model, data, params, mock_prediction_changed):
    # This test case requires different set-up and assertions, so we have it in separate function.
    metric_instance = metric(
        **params,
        disable_warnings=True,
        nr_samples=10,
        return_nan_when_prediction_changes=True,
    )
    result = metric_instance(
        model,
        data["x_batch"],
        data["y_batch"],
        explain_func=explain,
        explain_func_kwargs={
            "method": "Saliency",
        },
    )
    assert np.isnan(result).all()


@pytest.mark.robustness
def test_return_nan_when_prediction_changes_continuity(load_mnist_model, load_mnist_images, mock_prediction_changed):
    # Continuity returns dict, so we have it in separate function in order to keep assertions readable.
    metric_instance = Continuity(
        disable_warnings=True,
        return_nan_when_prediction_changes=True,
        nr_steps=10,
    )
    result = metric_instance(
        load_mnist_model,
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
        explain_func=explain,
        explain_func_kwargs={
            "method": "Saliency",
        },
    )
    for r in result:
        assert np.isnan(r).any()
