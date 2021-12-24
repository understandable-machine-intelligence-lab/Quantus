import pytest
from typing import Union
import torch
import torchvision
import pickle
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "params,expected",
    [
        ({"method": "Saliency", "img_size": 28, "nr_channels": 1}, {"min": 0}),
        ({"method": "Gradient", "img_size": 28, "nr_channels": 1, "pos_only": True}, {"min": 0}),
        ({"method": "Gradient", "img_size": 28, "nr_channels": 1, "abs": True}, {"min": 0}),
        ({"method": "Control Var. Constant", "img_size": 28, "nr_channels": 1, "constant_value": 0.0}, {"value": 0.0}),
        (
            {
                "method": "Gradient",
                "normalise": True,
                "normalise_func": normalise_by_negative,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            {
                "method": "Gradient",
                "normalise": True,
                "abs": True,
                "normalise_func": normalise_by_max,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_explain_func(
    params: dict, expected: Union[float, dict, bool], load_mnist_images, load_mnist_model
):
    a_batch = explain(
        model=load_mnist_model,
        inputs=load_mnist_images["x_batch"],
        targets=load_mnist_images["y_batch"],
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "min" in expected and "max" in expected:
            assert (a_batch.min() >= expected["min"]) & (
                a_batch.max() <= expected["max"]
            ), "Test failed."
        elif "min" in expected and "max" not in expected:
            assert a_batch.min() >= expected["min"], "Test failed."
        elif "min" not in expected and "max" in expected:
            assert a_batch.max() <= expected["min"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."
