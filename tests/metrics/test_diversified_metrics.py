from typing import Union, Dict
from pytest_lazyfixture import lazy_fixture
import pytest
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import gaussian_noise
from quantus.functions.similarity_func import correlation_spearman, distance_euclidean
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.diversified import MOON

@pytest.mark.moon
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
                    "nr_models": 5,
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
                    "nr_models": 5,
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
                    "nr_models": 10,
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
                    "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
                    "nr_models": 10,
                    "return_aggregate": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "DeepLift",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        #(
        #    lazy_fixture("load_mnist_model_tf"),
        #    lazy_fixture("load_mnist_images_tf"),
        #    {
        #        "init": {
        #            "noise_levels": [0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5],
        #            "nr_models": 3,
        #            "disable_warnings": True,
        #            "display_progressbar": True,
        #            "abs": True,
        #            "normalise": True,
        #        },
        #        "call": {
        #            "explain_func": explain,
        #            "explain_func_kwargs": {
#               "method": "VanillaGradients",
#                   },
#                },
        #            },
    #    {"min": 0.0, "max": 1.0},
        #     ),
    ],
)
def test_moon(
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
    scores = MOON(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    print(scores)
    if isinstance(expected, float):
        if np.isnan(expected):
            assert np.isnan(scores).any(), "Test failed."
        else:
            assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(
            ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
        ), "Test failed."

