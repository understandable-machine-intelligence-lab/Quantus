import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers.model_interface import ModelInterface


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "layer_order": "bottom_up",
                "similarity_func": correlation_pearson,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "layer_order": "top_down",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_model_parameter_randomisation(
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
    scores_layers = ModelParameterRandomisation(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    if isinstance(expected, float):
        assert all(
            s == expected for layer, scores in scores_layers.items() for s in scores
        ), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"]))
            for layer, scores in scores_layers.items()
            for s in scores
        ), "Test failed."


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "num_classes": 10,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_random_logit(
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
    scores = RandomLogit(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."
