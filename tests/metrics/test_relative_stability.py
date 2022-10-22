from __future__ import annotations

from pytest_lazyfixture import lazy_fixture  # noqa
from typing import Dict
import functools

from tests.fixtures import *
from quantus.functions.perturb_func import *
from quantus.functions.explanation_func import explain
from quantus.metrics.robustness import RelativeInputStability #RelativeRepresentationStability, RelativeOutputStability


# fmt: off
RIS_CONSTRUCTOR = functools.partial(RelativeInputStability,          nr_samples=5, disable_warnings=True)
#ROS_CONSTRUCTOR = functools.partial(RelativeOutputStability,         nr_samples=5, disable_warnings=True)
#RRS_CONSTRUCTOR = functools.partial(RelativeRepresentationStability, nr_samples=5, disable_warnings=True)
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
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_input_stability(model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs):

    ris = RIS_CONSTRUCTOR(**init_kwargs)
    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        reshape_input=False,
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
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini_batch"),
            {
                "perturb_func": gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {
                "perturb_func": gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "tf + mnist + perturb_func = gaussian_noise +  kwargs",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "torch + cifar10 + perturb_func = gaussian_noise + kwargs",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_output_stability(model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs):

    ris = ROS_CONSTRUCTOR(**init_kwargs)

    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        reshape_input=False,
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
        # MNIST
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_mnist"),
            lazy_fixture("load_mnist_images_tf_mini_batch"),
            {
                "perturb_func": gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
        ),
        # Cifar10
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images_mini_batch"),
            {
                "perturb_func": gaussian_noise,
                "perturb_func_kwargs": {
                    "perturb_std": 0.05,
                    "perturb_mean": 0.03,
                },
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {
                "normalise": True,
                "return_aggregate": True,
            },
            {},
        ),
        (
            lazy_fixture("load_cnn_2d_cifar"),
            lazy_fixture("load_cifar10_images_tf_mini_batch"),
            {},
            {"explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}},
        ),
    ],
    ids=[
        "tf + mnist + default perturb_func",
        "tf + mnist + perturb_func = gaussian_noise +  kwargs",
        "torch + mnist + normalise = True +  return_aggregate = True",
        "torch + mnist + method = IntegratedGradients",
        "torch + cifar10 + default perturb_func",
        "torch + cifar10 + perturb_func = gaussian_noise + kwargs",
        "tf + cifar10 + normalise = True + return_aggregate = True",
        "tf + cifar10 + method = GradCam",
    ],
)
def test_relative_representation_stability(model: tf.keras.Model, data: Dict[str, np.ndarray], init_kwargs, call_kwargs):

    ris = RRS_CONSTRUCTOR(**init_kwargs)

    x_batch = data["x_batch"]
    y_batch = predict(model, x_batch)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        explain_func=explain,
        reshape_input=False,
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]
