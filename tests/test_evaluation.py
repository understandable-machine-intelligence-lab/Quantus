import warnings
import pickle
from typing import Union

import numpy as np
import pytest
import torch
import torchvision
from pytest_lazyfixture import lazy_fixture

from .fixtures import *
from ..quantus import *
from ..quantus.helpers.explanation_func import explain
from ..quantus.helpers.pytorch_model import PyTorchModel


@pytest.mark.evaluate_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "normalise": True,
                "normalise_func": normalise_by_max,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "{params['method']: a_batch}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "normalise": True,
                "normalise_func": normalise_by_max,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "{params['method']: a_batch}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "nr_samples": 10,
                "abs": True,
                "explain_func": explain,
                "method": "IntegratedGradients",
                "disable_warnings": True,
                "normalise": False,
                "normalise_func": normalise_by_max,
                "eval_metrics": "{'sparseness': Sparseness(**params)}",
                "eval_xai_methods": "{params['method']: a_batch}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "normalise": False,
                "normalise_func": normalise_by_max,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "{params['method']: a_batch}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "normalise": False,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "[params['method']]",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "explain_func": explain,
                "method": "InputXGradient",
                "disable_warnings": True,
                "normalise": False,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "{params['method']: params['explain_func']}",
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "normalise": False,
                "eval_metrics": "None",
                "eval_xai_methods": "{params['method']: None}",
            },
            {"exception": TypeError},
        ),
    ],
)
def test_evaluate_func(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = data["x_batch"], data["y_batch"]
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            results = evaluate(
                metrics=eval(params["eval_metrics"]),
                xai_methods=eval(params["eval_xai_methods"]),
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                agg_func=np.mean,
                **params,
            )
        return

    results = evaluate(
        metrics=eval(params["eval_metrics"]),
        xai_methods=eval(params["eval_xai_methods"]),
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        agg_func=np.mean,
        **params,
    )

    if "min" in expected and "max" in expected:
        assert (
            results[params["method"]][list(eval(params["eval_metrics"]).keys())[0]]
            >= expected["min"]
        ), "Test failed."
        assert (
            results[params["method"]][list(eval(params["eval_metrics"]).keys())[0]]
            <= expected["max"]
        ), "Test failed."
