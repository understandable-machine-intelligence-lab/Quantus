import numpy as np
import pytest
import pickle
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers.tf_model import TensorFlowModel


@pytest.fixture
def mock_input_tf_array():
    return {"x": np.zeros((1, 28, 28, 1))}


@pytest.fixture
def flat_image_array():
    return {"x": np.zeros((1, 3*28*28)), "img_size": 28, "nr_channels": 3}


@pytest.mark.tf_model
@pytest.mark.parametrize("data,params,expected",
    [
        (
            lazy_fixture("mock_input_tf_array"),
            {"softmax_act": False, },
            np.array([-0.723556, 0.06658217, 0.13982001, -0.57502496, 0.19477458, 0.22203586,
                      -0.26914597, 0.23699084, -0.41618308, -0.5679564]),
        ),
        (
            lazy_fixture("mock_input_tf_array"),
            {"softmax_act": True, },
            np.array([0.05396363, 0.11891972, 0.12795599, 0.06260477, 0.13518457, 0.13892056,
                      0.08500589, 0.14101373, 0.07338234, 0.06304886]),
        ),
    ]
)
def test_predict(
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
        load_mnist_model_tf
):
    model = TensorFlowModel(load_mnist_model_tf)
    p = model.state_dict()
    model.load_state_dict(p)
    out = model.predict(
        x=data["x"],
        **params
    )
    assert np.allclose(out, expected), "Test failed."


@pytest.mark.tf_model
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("flat_image_array"), np.zeros((1, 28, 28, 3))),
    ],
)
def test_shape_input(
        data: np.ndarray,
        expected: Union[float, dict, bool],
        load_mnist_model_tf
):
    model = TensorFlowModel(load_mnist_model_tf)
    out = model.shape_input(**data)
    assert np.array_equal(out, expected), "Test failed."


'''
@pytest.mark.pytorch_model
@pytest.mark.parametrize("expected", [torch.nn.Module])
def test_get_model(
        expected: Union[float, dict, bool],
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    out = model.get_model()
    x = model.state_dict()
    assert isinstance(out, expected), "Test failed."


@pytest.mark.pytorch_model
@pytest.mark.parametrize("expected", [OrderedDict])
def test_state_dict(
        expected: Union[float, dict, bool],
        load_mnist_model
):
    model = PyTorchModel(load_mnist_model)
    out = model.state_dict()
    assert isinstance(out, expected), "Test failed."
'''
