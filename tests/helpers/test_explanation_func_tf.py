import pytest
from typing import Union
import torch
import torchvision
import pickle
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func_tf import explain_tf


@pytest.mark.explain_func_tf
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {"method": "Gradient", "img_size": 28, "nr_channels": 1, "pos_only": True},
            {"min": 0.0},
        ),
    ],
)
def test_explain_func_tf(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images_tf,
    load_mnist_model_tf,
):
    model = load_mnist_model_tf
    x_batch, y_batch = (
        load_mnist_images_tf["x_batch"],
        load_mnist_images_tf["y_batch"],
    )

    a_batch = explain_tf(
        model=model,
        inputs=x_batch,
        targets=y_batch,
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
            assert a_batch.max() <= expected["max"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."
