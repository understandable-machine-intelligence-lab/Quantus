from __future__ import annotations

from pytest_lazyfixture import lazy_fixture  # noqa
from typing import Dict, TYPE_CHECKING
import functools

if TYPE_CHECKING:
    import tensorflow as tf

from ..fixtures import *  # noqa
from ... import quantus

# Pre-fill kwargs which are always the sme while running tests
RIS_CONSTRUCTOR = functools.partial(quantus.RelativeInputStability, nr_samples=10, disable_warnings=True)
ROS_CONSTRUCTOR = functools.partial(quantus.RelativeOutputStability, nr_samples=10, disable_warnings=True)
RRS_CONSTRUCTOR = functools.partial(quantus.RelativeRepresentationStability, nr_samples=10, disable_warnings=True)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "x,xs, e_x, e_xs",
    [
        (
                np.random.random((32, 32, 1)),
                np.random.random((8, 32, 32, 1)),
                np.random.random((32, 32, 1)),
                np.random.random((8, 32, 32, 1))
        ),
        (
                np.random.random((32, 32, 3)),
                np.random.random((8, 32, 32, 3)),
                np.random.random((32, 32, 3)),
                np.random.random((8, 32, 32, 3))
        ),
        (
                np.random.random((1, 32, 32)),
                np.random.random((8, 1, 32, 32)),
                np.random.random((1, 32, 32)),
                np.random.random((8, 1, 32, 32))
        ),
        (
                np.random.random((3, 32, 32)),
                np.random.random((8, 3, 32, 32)),
                np.random.random((3, 32, 32)),
                np.random.random((8, 3, 32, 32))
        ),
    ],
    ids=[
        "1 channel,  channel last",
        "3 channels, channel last",
        "1 channel,  channel first",
        "3 channels, channel first"
    ],
)
def test_relative_input_stability_objective(x, xs, e_x, e_xs, capsys):
    with capsys.disabled():
        result = RIS_CONSTRUCTOR().relative_input_stability_objective(x=x, xs=xs, e_x=e_x, e_xs=e_xs)
        print(f"result = {result}")
    assert (result != np.nan).all()
    assert result.shape == (8,)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "h_x, h_xs, e_x, e_xs",
    [
        (
                np.random.random(10),
                np.random.random((8, 10)),
                np.random.random((32, 32, 1)),
                np.random.random((8, 32, 32, 1))
        ),
        (
                np.random.random(10),
                np.random.random((8, 10)),
                np.random.random((32, 32, 3)),
                np.random.random((8, 32, 32, 3))
        ),
        (
                np.random.random(10),
                np.random.random((8, 10)),
                np.random.random((1, 32, 32)),
                np.random.random((8, 1, 32, 32))
        ),
        (
                np.random.random(10),
                np.random.random((8, 10)),
                np.random.random((3, 32, 32)),
                np.random.random((8, 3, 32, 32))
        ),
    ],
    ids=[
        "1 channel,  channel last",
        "3 channels, channel last",
        "1 channel,  channel first",
        "3 channels, channel first"
    ],
)
def test_relative_output_stability_objective(h_x, h_xs, e_x, e_xs, capsys):
    with capsys.disabled():
        result = ROS_CONSTRUCTOR().relative_output_stability_objective(h_x=h_x, h_xs=h_xs, e_x=e_x, e_xs=e_xs)
        print(f"result = {result}")

    assert (result != np.nan).all()
    assert result.shape == (8,)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "l_x, l_xs, e_x, e_xs",
    [
        (
                np.random.random(256),
                np.random.random((8, 256)),
                np.random.random((32, 32, 1)),
                np.random.random((8, 32, 32, 1))
        ),
        (
                np.random.random(256),
                np.random.random((8, 256)),
                np.random.random((32, 32, 3)),
                np.random.random((8, 32, 32, 3))
        ),
        (
                np.random.random(256),
                np.random.random((8, 256)),
                np.random.random((1, 32, 32)),
                np.random.random((8, 1, 32, 32))
        ),
        (
                np.random.random(256),
                np.random.random((8, 256)),
                np.random.random((3, 32, 32)),
                np.random.random((8, 3, 32, 32))
        ),
    ],
    ids=[
        "1 channel,  channel last",
        "3 channels, channel last",
        "1 channel,  channel first",
        "3 channels, channel first"
    ],
)
def test_relative_representation_stability_objective(l_x, l_xs, e_x, e_xs, capsys):
    with capsys.disabled():
        result = RRS_CONSTRUCTOR().relative_representation_stability_objective(l_x=l_x, l_xs=l_xs, e_x=e_x, e_xs=e_xs)
        print(f"result = {result}")

    assert (result != np.nan).all()
    assert result.shape == (8,)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        # MNIST
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {

                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "IntegratedGradients"}
                }
        ),
        # Cifar10
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {

                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}
                }
        ),

    ],
    ids=[
        "mnist + default perturb_func",
        "mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "mnist + normalise = True +  return_aggregate = True",
        "mnist + method = IntegratedGradients",

        "cifar10 + default perturb_func",
        "cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "cifar10 + normalise = True + return_aggregate = True",
        "cifar10 + method = GradCam",
    ],
)
def test_relative_input_stability(
        model: tf.keras.Model,
        data: Dict[str, np.ndarray],
        init_kwargs,
        call_kwargs,
        capsys
):
    with capsys.disabled():
        ris = RIS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = model.predict(x_batch).argmax(axis=1)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs
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
                lazy_fixture("load_mnist_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {

                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "IntegratedGradients"}
                }
        ),
        # Cifar10
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {
                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}
                }
        ),

    ],
    ids=[
        "mnist + default perturb_func",
        "mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "mnist + normalise = True +  return_aggregate = True",
        "mnist + method = IntegratedGradients",

        "cifar10 + default perturb_func",
        "cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "cifar10 + normalise = True + return_aggregate = True",
        "cifar10 + method = GradCam",
    ],
)
def test_relative_output_stability(
        model: tf.keras.Model,
        data: Dict[str, np.ndarray],
        init_kwargs,
        call_kwargs,
        capsys
):
    with capsys.disabled():
        ris = ROS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = model.predict(x_batch).argmax(axis=1)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs
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
                lazy_fixture("load_mnist_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {

                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_mnist"),
                lazy_fixture("load_mnist_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "IntegratedGradients"}
                }
        ),
        # Cifar10
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {

                    "perturb_func": quantus.gaussian_noise,
                    "perturb_func_kwargs": {
                        "perturb_std": 0.05,
                        "perturb_mean": 0.03,
                    }
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {
                    "normalise": True,
                    "return_aggregate": True,
                },
                {}
        ),
        (
                lazy_fixture("load_cnn_2d_cifar"),
                lazy_fixture("load_cifar10_images_tf"),
                {},
                {
                    "explain_func_kwargs": {"method": "GradCam", "gc_layer": "test_conv"}
                }
        ),

    ],
    ids=[
        "mnist + default perturb_func",
        "mnist + perturb_func = quantus.gaussian_noise +  kwargs",
        "mnist + normalise = True +  return_aggregate = True",
        "mnist + method = IntegratedGradients",

        "cifar10 + default perturb_func",
        "cifar10 + perturb_func = quantus.gaussian_noise + kwargs",
        "cifar10 + normalise = True + return_aggregate = True",
        "cifar10 + method = GradCam",
    ],
)
def test_relative_representation_stability(
        model: tf.keras.Model,
        data: Dict[str, np.ndarray],
        init_kwargs,
        call_kwargs,
        capsys
):
    with capsys.disabled():
        ris = RRS_CONSTRUCTOR(**init_kwargs)

        x_batch = data["x_batch"]
        y_batch = model.predict(x_batch).argmax(axis=1)

        result = ris(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            explain_func=quantus.explain,
            reshape_input=False,
            **call_kwargs
        )
        result = np.asarray(result)
        print(f"result = {result}")

    assert (result != np.nan).all()

    if init_kwargs.get("return_aggregate", False):
        assert result.shape == (1,)
    else:
        assert result.shape[0] == x_batch.shape[0]