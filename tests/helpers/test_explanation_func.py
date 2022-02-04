import pytest
from typing import Union
import torch
import torchvision
import pickle
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Saliency", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "GradientShap", "img_size": 28, "nr_channels": 1, "abs": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "IntegratedGradients",
                "img_size": 28,
                "nr_channels": 1,
                "abs": True,
            },
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "InputXGradient", "img_size": 28, "nr_channels": 1, "abs": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Occlusion", "img_size": 28, "nr_channels": 1, "abs": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "FeatureAblation",
                "img_size": 28,
                "nr_channels": 1,
                "neg_only": True,
            },
            {"max": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "GradCam",
                "img_size": 28,
                "nr_channels": 1,
                "gc_layer": "model._modules.get('conv_2')",
                "abs": True,
            },
            {"min": 0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Sobel Filter",
                "img_size": 28,
                "nr_channels": 1,
                "neg_only": True,
            },
            {"max": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Gradient", "img_size": 28, "nr_channels": 1, "pos_only": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Gradient", "img_size": 28, "nr_channels": 1, "abs": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Constant",
                "img_size": 28,
                "nr_channels": 1,
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "normalise": True,
                "normalise_func": normalise_by_negative,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "normalise": True,
                "abs": True,
                "normalise_func": normalise_by_max,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Gradient",
                "abs": True,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Occlusion",
                "abs": True,
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "IntegratedGradients",
                "img_size": 28,
                "nr_channels": 1,
                "abs": True,
            },
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "InputXGradient", "img_size": 28, "nr_channels": 1, "abs": True},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"img_size": 28, "nr_channels": 1},
            {"warning": UserWarning},
        ),
        (
            None,
            lazy_fixture("load_mnist_images_tf"),
            {"img_size": 28, "nr_channels": 1},
            {"exception": ValueError},
        ),
    ],
)
def test_explain_func(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = explain(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **params,
            )
        return

    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "min" in expected and "max" in expected:
            assert (a_batch.min() >= expected["min"]) & (
                a_batch.max() <= expected["max"]
            ), "Test failed."
        elif "min" in expected and "max" not in expected:
            assert a_batch.min() >= expected["min"], "Test failed."
        elif "min" not in expected and "max" in expected:
            assert a_batch.max() <= expected["max"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."
        elif "warning" in expected:
            with pytest.warns(expected["warning"]):
                a_batch = explain(
                    model=model,
                    inputs=x_batch,
                    targets=y_batch,
                    **params,
                )


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Saliency", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Constant",
                "img_size": 28,
                "nr_channels": 1,
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "GradCam", "img_size": 28, "nr_channels": 1},
            {"exception": AssertionError},
        ),
    ],
)
def test_generate_captum_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = generate_captum_explanation(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **params,
            )
        return

    a_batch = generate_captum_explanation(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "min" in expected and "max" in expected:
            assert (a_batch.min() >= expected["min"]) & (
                a_batch.max() <= expected["max"]
            ), "Test failed."
        elif "min" in expected and "max" not in expected:
            assert a_batch.min() >= expected["min"], "Test failed."
        elif "min" not in expected and "max" in expected:
            assert a_batch.max() <= expected["max"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "Gradient", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "Occlusion", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "InputXGradient", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "IntegratedGradients", "img_size": 28, "nr_channels": 1},
            {"min": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "GradCam", "img_size": 28, "nr_channels": 1},
            {"exception": AssertionError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "GradCam",
                "img_size": 28,
                "nr_channels": 1,
                "gc_layer": "dense_1",
            },
            {"exception": Exception},
        ),
    ],
)
def test_generate_tf_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = generate_tf_explanation(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **params,
            )
        return

    a_batch = generate_tf_explanation(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "min" in expected and "max" in expected:
            assert (a_batch.min() >= expected["min"]) & (
                a_batch.max() <= expected["max"]
            ), "Test failed."
        elif "min" in expected and "max" not in expected:
            assert a_batch.min() >= expected["min"], "Test failed."
        elif "min" not in expected and "max" in expected:
            assert a_batch.max() <= expected["max"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "Gradient", "img_size": 28, "nr_channels": 1},
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {"method": "Gradient", "img_size": 28, "nr_channels": 1},
            {"min": -3},
        ),
    ],
)
def test_get_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    a_batch = get_explanation(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "min" in expected and "max" in expected:
            assert (a_batch.min() >= expected["min"]) & (
                a_batch.max() <= expected["max"]
            ), "Test failed."
        elif "min" in expected and "max" not in expected:
            assert a_batch.min() >= expected["min"], "Test failed."
        elif "min" not in expected and "max" in expected:
            assert a_batch.max() <= expected["max"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."
