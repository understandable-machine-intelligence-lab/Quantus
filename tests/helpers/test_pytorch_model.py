import torch
import numpy as np
import pytest
import pickle
from typing import Union
from functools import reduce
from operator import and_
from collections import OrderedDict
from pytest_lazyfixture import lazy_fixture
from scipy.special import softmax

from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers.pytorch_model import PyTorchModel


@pytest.fixture
def mock_input_torch_array():
    return {"x": np.zeros((1, 1, 28, 28))}


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": False,
                "device": "cpu",
            },
            np.array(
                [
                    -0.44321266,
                    0.60336196,
                    0.2091731,
                    -0.17474744,
                    -0.03755454,
                    0.5306321,
                    -0.3079375,
                    0.5329694,
                    -0.41116637,
                    -0.3060812,
                ]
            ),
        ),
        (
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": True,
                "device": "cpu",
            },
            softmax(
                np.array(
                    [
                        -0.44321266,
                        0.60336196,
                        0.2091731,
                        -0.17474744,
                        -0.03755454,
                        0.5306321,
                        -0.3079375,
                        0.5329694,
                        -0.41116637,
                        -0.3060812,
                    ]
                ),
            ),
        ),
        (
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": True,
                "device": "cpu",
                "training": True,
            },
            {"exception": AttributeError},
        ),
    ],
)
def test_predict(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    load_mnist_model.eval()
    training = params.pop("training", False)
    model = PyTorchModel(load_mnist_model, **params)
    if training:
        with pytest.raises(expected["exception"]):
            model.train()
            out = model.predict(x=data["x"])
        return
    out = model.predict(x=data["x"])
    assert np.allclose(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("flat_image_array"),
            {"channel_first": True},
            np.zeros((1, 3, 28, 28)),
        ),
        (
            lazy_fixture("flat_image_array"),
            {"channel_first": False},
            {"exception": ValueError},
        ),
        (
            lazy_fixture("flat_sequence_array"),
            {"channel_first": True},
            np.zeros((1, 3, 28)),
        ),
        (
            lazy_fixture("flat_sequence_array"),
            {"channel_first": False},
            {"exception": ValueError},
        ),
    ],
)
def test_shape_input(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    load_mnist_model.eval()
    model = PyTorchModel(load_mnist_model, channel_first=params["channel_first"])
    if not params["channel_first"]:
        with pytest.raises(expected["exception"]):
            out = model.shape_input(**data)
        return
    out = model.shape_input(**data)
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize("expected", [torch.nn.Module])
def test_get_model(expected: Union[float, dict, bool], load_mnist_model):
    model = PyTorchModel(load_mnist_model, channel_first=True)
    out = model.get_model()
    assert isinstance(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize("expected", [OrderedDict])
def test_state_dict(expected: Union[float, dict, bool], load_mnist_model):
    model = PyTorchModel(load_mnist_model, channel_first=True)
    out = model.state_dict()
    assert isinstance(out, expected), "Test failed."


@pytest.mark.pytorch_model
def test_get_random_layer_generator(load_mnist_model):
    model = PyTorchModel(load_mnist_model, channel_first=True)

    for layer_name, random_layer_model in model.get_random_layer_generator():

        layer = getattr(model.get_model(), layer_name).parameters()
        new_layer = getattr(random_layer_model, layer_name).parameters()

        assert layer != new_layer, "Test failed."
