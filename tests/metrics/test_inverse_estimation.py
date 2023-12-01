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
            {"min": 0.0, "max": 1.0},
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
            {"min": 0.0, "max": 1.0},
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
            {"min": 0.0, "max": 1.0},
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
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
            {"min": 0.0, "max": 1.0},
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
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
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
                    },
                },
            },
            {"min": 0.0, "max": 14.0},
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
            {"min": 0.0, "max": 10.0},
        ),
    ],
)
def test_pixel_flipping(
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

    metric_init = PixelFlipping(**init_params)
    inverse_estimation = InverseEstimation(metric_init=metric_init)
    scores = inverse_estimation(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    print("final scores!!!", scores[0][:10])
    print(
        "scores_inv!!!",
        (
            np.shape(inverse_estimation.scores_inv),
            inverse_estimation.scores_inv[0][:10],
        ),
    )
    print(
        "scores!!!",
        (np.shape(inverse_estimation.scores), inverse_estimation.scores[0][:10]),
    )

    assert all(
        [
            (s >= expected["min"] and s <= expected["max"])
            for s_list in scores
            for s in s_list
        ]
    ), "Test failed."
