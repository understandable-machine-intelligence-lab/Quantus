from collections import OrderedDict
from contextlib import nullcontext
from typing import Union
import sys

import numpy as np
import pytest
import pytest_mock
import torch
from pytest_lazyfixture import lazy_fixture
from scipy.special import softmax
from quantus.helpers.model.pytorch_model import PyTorchModel

def torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

@pytest.fixture
def mock_input_torch_array():
    return {"x": np.zeros((1, 1, 28, 28))}


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": False,
                "device": "cpu",
            },
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": True,
                "device": "cpu",
            },
        ),
        (
            lazy_fixture("load_mnist_model_softmax_not_last"),
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": False,
                "device": "cpu",
            },
        ),
        (
            lazy_fixture("load_mnist_model_softmax"),
            lazy_fixture("mock_input_torch_array"),
            {
                "softmax": False,
                "device": "cpu",
            },
        ),
    ],
)
def test_get_softmax_arg_model(
    model: torch.nn.Module,
    data: np.ndarray,
    params: dict,
):
    model.eval()

    sm_model = PyTorchModel(model, softmax=True)
    no_sm_model = PyTorchModel(model, softmax=False)

    sm_out = sm_model.predict(x=data["x"])
    no_sm_out = no_sm_model.predict(x=data["x"])

    assert np.allclose(sm_out, softmax(no_sm_out)), "Test failed."


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
def test_predict(data: np.ndarray, params: dict, expected: Union[float, dict, bool], load_mnist_model):
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
def test_shape_input(data: np.ndarray, params: dict, expected: Union[float, dict, bool], load_mnist_model):
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


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"layer_names": ["conv_2"]},
        {"layer_indices": [0, 1]},
        {"layer_indices": [-1, -2]},
    ],
    ids=["all layers", "2nd conv", "1st 2 layers", "last 2 layers"],
)
def test_get_hidden_layers_output(load_mnist_model, params):
    model = PyTorchModel(load_mnist_model, channel_first=True)
    X = np.random.random((32, 1, 28, 28))
    result = model.get_hidden_representations(X, **params)
    assert isinstance(result, np.ndarray), "Must be a np.ndarray"
    assert len(result.shape) == 2, "Must be a batch of 1D tensors"
    assert result.shape[0] == X.shape[0], "Must have same batch size as input"


@pytest.mark.pytorch_model
def test_add_mean_shift_to_first_layer(load_mnist_model):
    model = PyTorchModel(load_mnist_model, channel_first=True)
    shift = -1
    arr = np.random.random((32, 1, 28, 28))
    X = torch.Tensor(arr).to(model.device)

    X_shift = torch.Tensor(arr + shift).to(model.device)
    new_model = model.add_mean_shift_to_first_layer(shift, X.size())
    a1 = model.model(X)
    a2 = new_model(X_shift)
    assert torch.all(torch.isclose(a1, a2, atol=1e-04))


@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "hf_model,data,softmax,model_kwargs,expected",
    [
        (
            lazy_fixture("load_hf_distilbert_sequence_classifier"),
            lazy_fixture("dummy_hf_tokenizer"),
            False,
            {},
            nullcontext(np.array([[0.00424026, -0.03878461]])),
        ),
        (
            lazy_fixture("load_hf_distilbert_sequence_classifier"),
            lazy_fixture("dummy_hf_tokenizer"),
            False,
            {"labels": torch.tensor([1]), "output_hidden_states": True},
            nullcontext(np.array([[0.00424026, -0.03878461]])),
        ),
        (
            lazy_fixture("load_hf_distilbert_sequence_classifier"),
            {
                "input_ids": torch.tensor([[101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
            },
            False,
            {"labels": torch.tensor([1]), "output_hidden_states": True},
            nullcontext(np.array([[0.00424026, -0.03878461]])),
        ),
        (
            lazy_fixture("load_hf_distilbert_sequence_classifier"),
            lazy_fixture("dummy_hf_tokenizer"),
            True,
            {},
            nullcontext(np.array([[0.51075452, 0.4892454]])),
        ),
        (
            lazy_fixture("load_hf_distilbert_sequence_classifier"),
            np.array([1, 2, 3]),
            False,
            {},
            pytest.raises(ValueError),
        ),
    ],
)
def test_huggingface_classifier_predict(hf_model, data, softmax, model_kwargs, expected):
    model = PyTorchModel(model=hf_model, softmax=softmax, model_predict_kwargs=model_kwargs)
    with expected:
        out = model.predict(x=data)
        assert np.allclose(out, expected.enter_result, atol=1e-5), "Test failed."


@pytest.fixture
def mock_transformers_not_installed(mocker: pytest_mock.MockerFixture):
    mock_dict = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    mocker.patch.dict("sys.modules", mock_dict)
    model = mocker.MagicMock(spec=None)
    model.training = False
    yield model
    mocker.resetall(return_value=True, side_effect=True)


@pytest.mark.pytorch_model
def test_predict_transformers_not_installed(mock_transformers_not_installed):
    model = PyTorchModel(model=mock_transformers_not_installed, softmax=True)
    x = {"input_ids": np.array([1, 2, 3]), "attention_mask": np.array([1, 1, 1])}
    with pytest.raises(ValueError):
        model.predict(x)


@pytest.mark.pytorch_model
def test_predict_invalid_input(load_hf_distilbert_sequence_classifier):
    model = PyTorchModel(load_hf_distilbert_sequence_classifier)
    # Prepare input and call the predict method
    x = torch.tensor([1, 2, 3, 4])
    with pytest.raises(ValueError):
        model.predict(x)
