import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers.pytorch_model import PyTorchModel
from ...quantus.helpers.tf_model import TensorFlowModel


@pytest.mark.axiomatic
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            {
                "abs": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            {
                "abs": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            {
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": True,
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
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
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
    "params,expected",
    [
        (
            {
                "n_samples": 1,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            {
                "n_samples": 1,
                "eps": 1e-2,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            1.0,
        ),
        (
            {
                "n_samples": 2,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            1.0,
        ),
        (
            {
                "n_samples": 1,
                "eps": 1e-10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": True,
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
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
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
    "params,expected",
    [
        (
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputxGradient",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": True,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"dtypes": [True, False]},
        ),
        (
            {
                "abs": False,
                "normalise": False,
                "explain_func": explain,
                "method": "InputXGradient",
                "img_size": 28,
                "nr_channels": 1,
                "input_shift": -1,
                "disable_warnings": True,
                "display_progressbar": True,
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
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
    )

    scores = InputInvariance(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        **params,
    )

    assert np.all([s in expected["dtypes"] for s in scores]), "Test failed."
