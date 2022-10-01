from __future__ import annotations


import numpy as np
import torch
from pytest_lazyfixture import lazy_fixture  # noqa
from typing import Dict
import functools
import tensorflow as tf

from ..fixtures import *  # noqa
from ... import quantus

# fmt: off
RIS_CONSTRUCTOR = functools.partial(quantus.RelativeInputStability,          nr_samples=5, disable_warnings=True)
ROS_CONSTRUCTOR = functools.partial(quantus.RelativeOutputStability,         nr_samples=5, disable_warnings=True)
RRS_CONSTRUCTOR = functools.partial(quantus.RelativeRepresentationStability, nr_samples=5, disable_warnings=True)
# fmt: on


def predict(model: tf.keras.Model| torch.nn.Module, x_batch: np.ndarray) -> np.ndarray:
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            return model(torch.Tensor(x_batch)).argmax(axis=1).numpy()
    else:
        return model.predict(x_batch).argmax(1)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "tf + mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "torch + cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_input_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs, capsys
):
    with capsys.disabled():
        ris = RIS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = predict(model, x_batch)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs,
        )
        result = np.asarray(result)
        print(f"result = {result}")

    assert (result != np.nan).all()

    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "tf + mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "torch + cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_output_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs, capsys
):
    with capsys.disabled():
        ris = ROS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = predict(model, x_batch)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs,
        )
        result = np.asarray(result)
        print(f"result = {result}")

    assert (result != np.nan).all()

    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini"),
            {
                "perturb_func": quantus.gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "tf + mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "torch + cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_representation_stability(
    model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs, capsys
):
    with capsys.disabled():
        ris = RRS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = predict(model, x_batch)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs,
        )
        result = np.asarray(result)
        print(f"result = {result}")

    assert (result != np.nan).all()

    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]
