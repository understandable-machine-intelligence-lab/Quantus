from functools import reduce
from operator import and_
from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from scipy.special import softmax


from quantus.helpers.model.tf_model import TensorFlowModel

EXPECTED_LOGITS = np.array(
    [
        -0.723556,
        0.06658217,
        0.13982001,
        -0.57502496,
        0.19477458,
        0.22203586,
        -0.26914597,
        0.23699084,
        -0.41618308,
        -0.5679564,
    ]
)


@pytest.mark.tf_model
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            np.zeros((1, 28, 28, 1)),
            {
                "softmax": False,
                "channel_first": False,
            },
            EXPECTED_LOGITS,
        ),
        (
            np.zeros((1, 28, 28, 1)),
            {
                "softmax": True,
                "channel_first": False,
            },
            softmax(EXPECTED_LOGITS),
        ),
    ],
)
def test_predict(
    data: np.ndarray, params: dict, expected: np.ndarray, load_mnist_model_tf, mocker
):
    mocker.patch(
        "tensorflow.keras.Model.predict", lambda x, *args, **kwargs: EXPECTED_LOGITS
    )
    model = TensorFlowModel(model=load_mnist_model_tf, **params)
    out = model.predict(x=data)
    assert np.allclose(out, expected), "Test failed."


@pytest.mark.tf_model
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("flat_image_array"),
            {"channel_first": False},
            np.zeros((1, 28, 28, 3)),
        ),
        (
            lazy_fixture("flat_image_array"),
            {"channel_first": True},
            np.zeros((1, 3, 28, 28)),
        ),
        (
            lazy_fixture("flat_sequence_array"),
            {"channel_first": False},
            np.zeros((1, 28, 3)),
        ),
        (
            lazy_fixture("flat_sequence_array"),
            {"channel_first": True},
            np.zeros((1, 3, 28)),
        ),
    ],
)
def test_shape_input(
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_model_tf,
):
    model = TensorFlowModel(load_mnist_model_tf, channel_first=params["channel_first"])
    out = model.shape_input(**data)
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.tf_model
def test_get_random_layer_generator(load_mnist_model_tf):
    tf_model = load_mnist_model_tf
    model = TensorFlowModel(model=tf_model, channel_first=False)
    before = model.state_dict()
    old_weights = {s.name: s.get_weights() for s in list(tf_model.layers)}

    for layer_name, random_layer_model in model.get_random_layer_generator():

        old = old_weights[layer_name]
        new = random_layer_model.get_layer(layer_name).get_weights()

        assert reduce(
            and_, [not np.allclose(x, y) for x, y in zip(old, new)]
        ), "Test failed."

    after = model.state_dict()

    # Make sure the original model is unaffected
    assert reduce(
        and_, [np.allclose(x, y) for x, y in zip(before, after)]
    ), "Test failed."



@pytest.mark.tf_model
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"layer_names": ["test_conv"]},
        {"layer_indices": [0, 1]},
        {"layer_indices": [-1, -2]},
    ],
    ids=["all layers", "2nd conv", "first 2 layers", "last 2 layers"],
)
def test_get_hidden_layer_output_sequential(load_mnist_model_tf, params):
    model = TensorFlowModel(model=load_mnist_model_tf, channel_first=False)
    X = np.random.random((32, 28, 28, 1))
    result = model.get_hidden_representations(X, **params)
    assert isinstance(result, np.ndarray), "Must be a np.ndarray"
    assert len(result.shape) == 2, "Must be a batch of 1D tensors"
    assert result.shape[0] == X.shape[0], "Must have same batch size as input"


@pytest.mark.tf_model
def test_add_mean_shift_to_first_layer(load_mnist_model_tf):
    model = TensorFlowModel(model=load_mnist_model_tf, channel_first=False)
    shift = 0.2
    X = np.random.random((32, 28, 28, 1))
    X_shift = X + shift
    new_model = model.add_mean_shift_to_first_layer(shift, X[:1].shape)
    a1 = model.model(X)
    a2 = new_model(X_shift)
    assert np.all(np.isclose(a1, a2, atol=1e-04))
