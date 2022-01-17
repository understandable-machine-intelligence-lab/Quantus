import numpy as np
import pytest
import pickle
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers.pytorch_model import PyTorchModel
import torch
from collections import OrderedDict


@pytest.fixture
def mock_input_torch_array():
    return {"x": np.zeros((1, 1, 28, 28))}


@pytest.fixture
def flat_image_array():
    return {"x": np.zeros((1, 3*28*28)), "img_size": 28, "nr_channels": 3}


@pytest.mark.pytorch_model
@pytest.mark.parametrize("data,params,expected",
    [
        (
            lazy_fixture("mock_input_torch_array"),
            {"softmax_act": False, "device": "cpu", },
            np.array([-0.44321266, 0.60336196, 0.2091731, -0.17474744, -0.03755454,
                      0.5306321, -0.3079375, 0.5329694, -0.41116637, -0.3060812]),
        ),
        (
            lazy_fixture("mock_input_torch_array"),
            {"softmax_act": True, "device": "cpu", },
            np.array([0.05812924, 0.16554506, 0.1116149,  0.07603046, 0.08721069, 0.1539324,
                      0.06654936, 0.15429261, 0.06002224, 0.06667302]),
        ),
    ]
)
def test_predict(
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    out = model.predict(
        x=data["x"],
        **params
    )
    assert np.allclose(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("flat_image_array"), np.zeros((1, 3, 28, 28))),
    ],
)
def test_shape_input(
        data: np.ndarray,
        expected: Union[float, dict, bool],
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    out = model.shape_input(**data)
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize("expected", [list])
def test_get_layers(
        expected: Union[float, dict, bool],
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    out = model.get_layers()
    assert isinstance(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize("params",
    [
        ({"layer_name": "conv_2"}),
    ],
)
def test_randomize_layer(
        params: dict,
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    layer = getattr(model.get_model(), params["layer_name"]).parameters()
    model.randomize_layer(**params)
    new_layer = getattr(model.get_model(), params["layer_name"]).parameters()
    assert (layer != new_layer), "Test failed."
