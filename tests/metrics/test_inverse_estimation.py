from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import (
    baseline_replacement_by_indices,
    noisy_linear_imputation,
)
from quantus.metrics.inverse_estimation import InverseEstimation
from quantus.metrics.faithfulness import (
    PixelFlipping,
    RegionPerturbation,
)


@pytest.mark.inverse_estimation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
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
            {"min": -1000.0, "max": 1000.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 14,
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
            {"min": -1000.0, "max": 1000.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "uniform",
                    "features_in_step": 56,
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
            {"min": -1000.0, "max": 1000.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "uniform",
                    "features_in_step": 112,
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
            {"exception": AssertionError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
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
            {"min": -1000.0, "max": 1000.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "features_in_step": 10,
                    "normalise": False,
                    "perturb_func": baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                "call": {},
            },
            {"exception": AssertionError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "uniform",
                    "features_in_step": 56,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                        "softmax": False,
                    },
                },
            },
            {"min": -1000.0, "max": 1000.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": True,
                "init": {
                    "features_in_step": 10,
                    "normalise": False,
                    "perturb_func": baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "xai_lib": "captum",
                    },
                    "softmax": False,
                },
            },
            {"exception": AssertionError},
        ),
    ],
)
def test_inverse_estimation_with_pixel_flipping(
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

    if "a_batch" in data:
        a_batch = data["a_batch"]
    elif params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
        assert a_batch is not None
    else:
        a_batch = None

    metric_init = PixelFlipping(**init_params)
    metric_init.softmax = True

    try:
        inv = InverseEstimation(metric_init=metric_init, return_aggregate=True)
        scores = inv(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            **call_params,
        )
        print(f"\n\n\tscores: {np.shape(inv.scores)},\n{inv.scores}")
        print(f"\n\n\tscores_inv: {np.shape(inv.scores_inv)},\n{inv.scores_inv}")
        print(
            f"\n\n\tall_evaluation_scores: {np.shape(inv.all_evaluation_scores)},\n{inv.all_evaluation_scores}"
        )
        print(f"\n\n\tscores: {np.shape(scores)},\n{scores}")

        if "exception" not in expected:
            assert all(
                [
                    (s >= expected["min"] and s <= expected["max"])
                    for s_list in scores
                    for s in s_list
                ]
            ), "Test failed."
    except expected["exception"] as e:
        print(f'Raised exception type {expected["exception"]}', e)
        return
