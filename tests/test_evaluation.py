import pytest
from typing import Union
import torch
import torchvision
import pickle
from pytest_lazyfixture import lazy_fixture
from .fixtures import *
from ..quantus import *

# from ..quantus.helpers import *
# from ..quantus.metrics import *


@pytest.mark.evaluate_func
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
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
            {
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
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
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
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
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "InputXGradient",
                "disable_warnings": True,
                "normalise": False,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**params)}",
                "eval_xai_methods": "{params['method']: params['explain_func']}",
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_evaluate_func(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images,
    load_mnist_model,
):
    model = load_mnist_model
    x_batch, y_batch = (
        load_mnist_images["x_batch"].numpy(),
        load_mnist_images["y_batch"].numpy(),
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    results = evaluate(
        metrics=eval(params["eval_metrics"]),
        xai_methods=eval(params["eval_xai_methods"]),
        model=load_mnist_model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        agg_func=np.mean,
        **params,
    )

    assert (
        results[params["method"]][list(eval(params["eval_metrics"]).keys())[0]]
        >= expected["min"]
    ), "Test failed."
    assert (
        results[params["method"]][list(eval(params["eval_metrics"]).keys())[0]]
        <= expected["max"]
    ), "Test failed."
