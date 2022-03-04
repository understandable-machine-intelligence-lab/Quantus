import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers.pytorch_model import PyTorchModel
from ...quantus.helpers.tf_model import TensorFlowModel


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "normalise": True,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "normalise": True,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
                "a_batch_generate": False,
            },
            1.0,
        ),
    ],
)
def test_completeness(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    explain = params["explain_func"]
    if params.get("a_batch_generate", True):
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    else:
        a_batch = None
    scores = Completeness(**params)(
        model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, **params
    )

    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "normalise": True,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "n_samples": 1,
                "normalise": True,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "n_samples": 2,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "n_samples": 2,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
            },
            1.0,
        ),
    ],
)
def test_non_sensitivity(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    explain = params["explain_func"]
    if params.get("a_batch_generate", True):
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    else:
        a_batch = None
    scores = NonSensitivity(**params)(
        model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, **params
    )
    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "InputxGradient",
                "img_size": (28, 28),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": False,
                "explain_func": explain,
                "method": "InputxGradient",
                "img_size": (28, 30),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": (28, 28),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": (28, 30),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 28),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_unequal_height_and_width"),
            {
                "abs": True,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": (28, 30),
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
    ],
)
def test_inputinvariance(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    scores = InputInvariance(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        **params,
    )

    assert np.all([s in expected["dtypes"] for s in scores]), "Test failed."
