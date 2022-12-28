from __future__ import annotations

import pytest
import torch
from pytest_lazyfixture import lazy_fixture
from typing import Dict, TYPE_CHECKING
import functools

from quantus.functions.perturb_func import *
from quantus.functions.explanation_func import explain
from quantus.metrics.robustness import (
    RelativeInputStability,
    RelativeOutputStability,
    RelativeRepresentationStability,
)

if TYPE_CHECKING:
    import tensorflow as tf

# fmt: off
RIS_CONSTRUCTOR = functools.partial(RelativeInputStability, nr_samples=5, disable_warnings=True)
ROS_CONSTRUCTOR = functools.partial(RelativeOutputStability, nr_samples=5, disable_warnings=True)
RRS_CONSTRUCTOR = functools.partial(RelativeRepresentationStability, nr_samples=5, disable_warnings=True)


# fmt: on


def predict(model: tf.keras.Model | torch.nn.Module, x_batch: np.ndarray) -> np.ndarray:
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            return model(torch.Tensor(x_batch)).argmax(axis=1).numpy()
    else:
        return model.predict(x_batch, verbose=0).argmax(1)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "GradientShap"}},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "return_aggregate": True,
                "return_nan_when_prediction_changes": False,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {},
            {},
        ),
    ],
)
def test_relative_input_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs
):
    ris = RIS_CONSTRUCTOR(**init_kwargs)
    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "GradientShap"}},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "return_aggregate": True,
                "return_nan_when_prediction_changes": False,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {},
            {},
        ),
    ],
)
def test_relative_output_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs
):
    ris = ROS_CONSTRUCTOR(**init_kwargs)

    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "GradientShap"}},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "return_aggregate": True,
                "return_nan_when_prediction_changes": False,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {},
            {},
        ),
    ],
)
def test_relative_representation_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs
):
    ris = RRS_CONSTRUCTOR(**init_kwargs)

    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric",
    [RIS_CONSTRUCTOR, ROS_CONSTRUCTOR, RRS_CONSTRUCTOR],
    ids=["RIS", "ROS", "RRS"],
)
def test_return_nan(metric, load_mnist_model_tf, load_mnist_images_tf):
    x_batch = load_mnist_images_tf["x_batch"]
    y_batch = predict(load_mnist_model_tf, x_batch)

    rs = metric(perturb_func_kwargs={"upper_bound": 255, "lower_bound": -255})
    result = rs(
        model=load_mnist_model_tf,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
    )
    assert np.isnan(result).any(), "Test Failed"
