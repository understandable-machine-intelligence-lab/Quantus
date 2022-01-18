import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "abs": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "abs": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
    ],
)
def test_completeness(
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
    scores = Completeness(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
    )

    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "n_samples": 1,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
        (
            {
                "n_samples": 2,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            1.0,
        ),
    ],
)
def test_non_sensitivity(
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
    scores = NonSensitivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
    )
    assert scores is not None, "Test failed."


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "InputxGradient",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
            },
            {"dtypes": [True, False]},
        ),
    ],
)
def test_inputinvariance(
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

    scores = InputInvariance(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        **params,
    )

    assert np.all([s in expected["dtypes"] for s in scores]), "Test failed."
