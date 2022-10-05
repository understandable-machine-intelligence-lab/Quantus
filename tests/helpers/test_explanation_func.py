import pickle
from typing import Union

import numpy as np
import torch
import torchvision
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *

# This is not nice
if util.find_spec("zennit"):
    from zennit import canonizers as zcanon
    from zennit import composites as zcomp
    from zennit import attribution as zattr
    from zennit import core as zcore
    from zennit import torchvision as ztv

if util.find_spec("zennit"):
    zennit_tests = [
        # Zennit
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "canonizer": None,
                "composite": None,
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": None,
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "canonizer": ztv.SequentialMergeBatchNorm,
                "composite": zcomp.EpsilonPlus,
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": ztv.SequentialMergeBatchNorm,
                "composite": zcomp.EpsilonPlus,
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "canonizer": None,
                "composite": "epsilon_alpha2_beta1_flat",
                "attributor": zattr.Gradient,
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "epsilon_alpha2_beta1_flat",
                "attributor": zattr.Gradient,
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
                "reduce_axes": (1,),
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
                "reduce_axes": (1, 2),
            },
            {"shape": (124, 1, 1, 28)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
                "reduce_axes": (3,),
            },
            {"shape": (124, 1, 28, 1)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
                "reduce_axes": (0, 1),
            },
            {"exception": AssertionError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "canonizer": None,
                "composite": "guided_backprop",
                "attributor": zattr.Gradient,
                "xai_lib": "zennit",
                "reduce_axes": (1, 2, 3, 4, 5),
            },
            {"exception": AssertionError},
        ),
    ]
else:
    zennit_tests = []


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    zennit_tests
    + [
        # Captum
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Saliency",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Saliency",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "GradientShap",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "GradientShap",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "IntegratedGradients",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "IntegratedGradients",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "InputXGradient",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "InputXGradient",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Occlusion",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Occlusion",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "FeatureAblation",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "FeatureAblation",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "GradCam",
                "gc_layer": "model._modules.get('conv_2')",
            },
            {"shape": (10, 1, 44)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "GradCam",
                "gc_layer": "model._modules.get('conv_2')",
            },
            {"shape": (124, 1, 8, 8)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Control Var. Sobel Filter",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Sobel Filter",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Gradient",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Control Var. Constant",
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Constant",
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Random Uniform",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Control Var. Random Uniform",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "reduce_axes": (1,),
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "normalise": True,
                "abs": True,
                "normalise_func": normalise_by_max,
                "reduce_axes": (1, 2),
            },
            {"shape": (124, 1, 1, 28)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "reduce_axes": (3,),
            },
            {"shape": (124, 1, 28, 1)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "reduce_axes": (0, 1),
            },
            {"exception": AssertionError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
                "reduce_axes": (1, 2, 3, 4, 5, 6),
            },
            {"exception": AssertionError},
        ),
        # tf-explain
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Gradient",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Occlusion",
            },
            {"shape": (124, 28, 28, 3)},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "IntegratedGradients",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "InputXGradient",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"warning": UserWarning},
        ),
        (
            None,
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "InputXGradient", "reduce_axes": (0, 1, 2)},
            {"exception": KeyError},
        ),
    ],
)
def test_explain_func(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):

    x_batch, y_batch = (data["x_batch"], data["y_batch"])
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = explain(model=model, inputs=x_batch, targets=y_batch, **params)
        return

    a_batch = explain(model=model, inputs=x_batch, targets=y_batch, **params)

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
        elif "shape" in expected:
            assert a_batch.shape == expected["shape"], "Test failed."
        elif "warning" in expected:
            with pytest.warns(expected["warning"]):
                a_batch = explain(
                    model=model, inputs=x_batch, targets=y_batch, **params
                )


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Saliency",
            },
            {"shape": (10, 1, 100)},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Saliency",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Control Var. Constant",
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Control Var. Constant",
                "constant_value": 0.0,
            },
            {"value": 0.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "GradCam",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "GradCam",
            },
            {"exception": ValueError},
        ),
    ],
)
def test_generate_captum_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (data["x_batch"], data["y_batch"])

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = generate_captum_explanation(
                model=model, inputs=x_batch, targets=y_batch, **params
            )
        return

    a_batch = generate_captum_explanation(
        model=model, inputs=x_batch, targets=y_batch, **params
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
        elif "shape" in expected:
            assert a_batch.shape == expected["shape"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."


@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "Gradient",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Gradient",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "Occlusion",
            },
            {"exception": IndexError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "Occlusion",
            },
            {"shape": (124, 28, 28, 3)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "InputXGradient",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "InputXGradient",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "IntegratedGradients",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "IntegratedGradients",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "GradCam",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "GradCam",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "GradCam",
                "gc_layer": "dense_1",
            },
            {"exception": Exception},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "method": "GradCam",
                "gc_layer": "dense_1",
            },
            {"exception": ValueError},
        ),
    ],
)
def test_generate_tf_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (data["x_batch"], data["y_batch"])

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = generate_tf_explanation(
                model=model, inputs=x_batch, targets=y_batch, **params
            )
        return

    a_batch = generate_tf_explanation(
        model=model, inputs=x_batch, targets=y_batch, **params
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
        elif "shape" in expected:
            assert a_batch.shape == expected["shape"], "Test failed."
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
            {
                "method": "Gradient",
            },
            {"shape": (124, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "method": "Gradient",
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "method": "Gradient",
            },
            {"shape": (124, 1, 28, 28)},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "method": "Gradient",
            },
            {"shape": (10, 1, 100)},
        ),
    ],
)
def test_get_explanation(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = data["x_batch"], data["y_batch"]

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            a_batch = get_explanation(
                model=model, inputs=x_batch, targets=y_batch, **params
            )
        return

    a_batch = get_explanation(model=model, inputs=x_batch, targets=y_batch, **params)

    if isinstance(expected, float):
        assert all(s == expected for s in a_batch), "Test failed."
    else:
        if "shape" in expected:
            assert a_batch.shape == expected["shape"], "Test failed."
        elif "value" in expected:
            assert all(
                s == expected["value"] for s in a_batch.flatten()
            ), "Test failed."
