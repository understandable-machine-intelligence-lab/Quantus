import numpy as np
import pytest
from functools import reduce
from operator import and_
from typing import Union
from scipy.special import softmax
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers.tf_model import TensorFlowModel


@pytest.fixture
def mock_input_tf_array():
    return {"x": np.zeros((1, 28, 28, 1))}


@pytest.fixture
def flat_image_array():
    return {"x": np.zeros((1, 3 * 28 * 28)), "img_size": 28, "nr_channels": 3}


@pytest.mark.tf_model
@pytest.mark.parametrize(
    "data,params,expected",
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
            softmax(
                np.array([-0.723556, 0.06658217, 0.13982001, -0.57502496, 0.19477458, 0.22203586,
                          -0.26914597, 0.23699084, -0.41618308, -0.5679564]),
            )
        ),
    ]
)
def test_predict(
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_model_tf,
):
    model = TensorFlowModel(load_mnist_model_tf)
    out = model.predict(x=data["x"], **params)
    assert np.allclose(out, expected), "Test failed."


@pytest.mark.tf_model
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("flat_image_array"), np.zeros((1, 28, 28, 3))),
    ],
)
def test_shape_input(
    data: np.ndarray, expected: Union[float, dict, bool], load_mnist_model_tf
):
    model = TensorFlowModel(load_mnist_model_tf)
    out = model.shape_input(**data)
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.tf_model
def test_get_random_layer_generator(load_mnist_model_tf):
    tf_model = load_mnist_model_tf
    model = TensorFlowModel(tf_model)
    old_weights = {s.name: s.get_weights() for s in list(tf_model.layers)}

    for layer_name, random_layer_model in model.get_random_layer_generator():

        old = old_weights[layer_name]

        new = random_layer_model.get_layer(layer_name).get_weights()

        assert reduce(
            and_, [not np.allclose(x, y) for x, y in zip(old, new)]
        ), "Test failed."


@pytest.mark.tf_model
def test_load_state_dict(load_mnist_model_tf):
    tf_model = load_mnist_model_tf
    model = TensorFlowModel(tf_model)
    before = model.state_dict()

    alayer = [l for l in tf_model.layers if len(l.get_weights()) > 0][0]
    weights = alayer.get_weights()
    alayer.set_weights([np.random.permutation(w) for w in weights])

    model.load_state_dict(before)
    after = model.state_dict()

    assert reduce(
        and_, [np.allclose(x, y) for x, y in zip(before, after)]
    ), "Test failed."
