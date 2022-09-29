from __future__ import annotations
import numpy as np
from pytest_lazyfixture import lazy_fixture  # noqa

from tests.fixtures import *  # noqa
import quantus
from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    import tensorflow as tf


"""
Following scenarios are to be tested for each Relative Stability metric

    - RS objective: 
            - 1 channel
            - 3 channels

    - Pre-computed perturbations
    - Pre-computed perturbations shape = x_batch.shape
    - Different perturb functions

    - Pre-computed explanations
    - only a_batch or as_batch given
    - Pre-computed explanations perturbed.shape = Pre-computed explanations shape
    - Different XAI methods
    
    - return_aggregate True/False
    - abs True/False
    - normalize True/False
    
"""


@pytest.mark.robustness
@pytest.mark.parametrize(
    "x,xs",
    [
        (np.random.random((10, 32, 32, 1)), np.random.random((10, 32, 32, 1))),
        (np.random.random((10, 32, 32, 3)), np.random.random((10, 32, 32, 3))),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_input_stability_objective(x, xs, capsys):
    result = quantus.RelativeInputStability().relative_input_stability_objective(
        x, xs, x, xs
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert (
            result.shape[0] == x.shape[0]
    ), "Must output same dimension as inputs batch axis"


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
                        "indices": list(range(124)),
                        "indexed_axes": [0],
                        "perturb_std": 0.5,
                        "perturb_mean": 0.3,
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
                        "indices": list(range(124)),
                        "indexed_axes": [0],
                        "perturb_std": 0.5,
                        "perturb_mean": 0.3,
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
        "mnist -> default perturb_func",
        "mnist -> perturb_func = quantus.gaussian_noise, with extra kwargs",
        "mnist -> normalise = True +  return_aggregate = True",
        "mnist -> method = IntegratedGradients",

        "cifar10 -> default perturb_func",
        "cifar10 -> perturb_func = quantus.gaussian_noise, with extra kwargs",
        "cifar10 -> normalise = True + return_aggregate = True",
        "cifar10 -> method = GradCam",
    ],
)
def test_relative_input_stability(
        model: tf.keras.Model,
        data: Dict[str, np.ndarray],
        init_kwargs,
        call_kwargs,
        capsys
):
    ris = quantus.RelativeInputStability(nr_samples=10, **init_kwargs)
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
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    if init_kwargs.get("return_aggreagte", False):
        assert len(result) == 0
    else:
        assert len(result) == len(x_batch), "Must have same batch size"


"""
@pytest.mark.robustness
@pytest.mark.parametrize(
    "h,hs,a,a_s",
    [
        (
            np.random.random((5, 10)),
            np.random.random((5, 10)),
            np.random.random((5, 32, 32, 1)),
            np.random.random((5, 32, 32, 1)),
        ),
        (
            np.random.random((5, 10)),
            np.random.random((5, 10)),
            np.random.random((5, 32, 32, 3)),
            np.random.random((5, 32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_output_stability_objective(h, hs, a, a_s, capsys):
    result = quantus.RelativeOutputStability.relative_output_stability_objective(
        h, hs, a, a_s
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert (
        result.shape[0] == h.shape[0]
    ), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "lx,lxs,a,a_s",
    [
        (
            np.random.random((5, 128)),
            np.random.random((5, 128)),
            np.random.random((5, 32, 32, 1)),
            np.random.random((5, 32, 32, 1)),
        ),
        (
            np.random.random((5, 128)),
            np.random.random((5, 128)),
            np.random.random((5, 32, 32, 3)),
            np.random.random((5, 32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_representation_stability_objective(lx, lxs, a, a_s, capsys):
    result = quantus.RelativeRepresentationStability.relative_representation_stability_objective(
        lx, lxs, a, a_s
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert (
        result.shape[0] == lx.shape[0]
    ), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params",
    [
        # no explain func, no pre computed explanations
        {},
        # only a_batch given
        {"a_batch": np.random.random((124, 28, 28, 1))},
        # pre-computed perturbed explanations have no extra batch axis
        {
            "a_batch": np.random.random((124, 28, 28, 1)),
            "as_batch": np.random.random((124, 28, 28, 1)),
        },
        # provided pre-computed perturbed explanations, but not perturbed x
        {
            "a_batch": np.random.random((124, 28, 28, 1)),
            "as_batch": np.random.random((5, 124, 28, 28, 1)),
        },
    ],
    ids=[
        "no explain func, no pre computed explanations",
        "pre-computed perturbations don't have extra batch dimension",
        "only a_batch given",
        "only as_batch given",
        "pre-computed perturbed explanations have no extra batch axis",
        "provided pre-computed perturbed explanations, but not perturbed x",
    ],
)
def test_invalid_kwargs(load_cnn_2d_1channel_tf, load_mnist_images_tf, params):
    with pytest.raises(ValueError):
        ris = quantus.RelativeInputStability()
        ris(
            load_cnn_2d_1channel_tf,
            load_mnist_images_tf["x_batch"],
            load_mnist_images_tf["y_batch"],
            **params,
        )
"""

"""
@pytest.mark.robustness
def test_relative_output_stability(
    load_cnn_2d_1channel_tf, load_mnist_images_tf, capsys
):
    ros = quantus.RelativeOutputStability()

    result = ros(
        load_cnn_2d_1channel_tf,
        load_mnist_images_tf["x_batch"],
        load_mnist_images_tf["y_batch"],
        explain_func=quantus.explain,
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert (
        result.shape[0] == load_mnist_images_tf["x_batch"].shape[0]
    ), "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params",
    [
        {"explain_func": quantus.explain},
        {"explain_func": quantus.explain, "layer_names": ["test_conv"]},
        {"explain_func": quantus.explain, "layer_indices": [7, 8]},
    ],
    ids=["2d CNN + mnist", "conv layer only", "last 2 layers"],
)
def test_relative_representation_stability(
    load_cnn_2d_1channel_tf, load_mnist_images_tf, params, capsys
):
    rrs = quantus.RelativeRepresentationStability()

    result = rrs(
        load_cnn_2d_1channel_tf,
        load_mnist_images_tf["x_batch"],
        load_mnist_images_tf["y_batch"],
        **params,
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert (
        result.shape[0] == load_mnist_images_tf["x_batch"].shape[0]
    ), "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric, params",
    [
        (
            quantus.RelativeInputStability,
            {"explain_func": quantus.explain},
        ),
        (
            quantus.RelativeOutputStability,
            {"explain_func": quantus.explain},
        ),
        (
            quantus.RelativeRepresentationStability,
            {"explain_func": quantus.explain},
        ),
    ],
    ids=["RIS", "ROS", "RRS"],
)
def test_relative_stability_pytorch(
    load_mnist_model, load_mnist_images, metric, params, capsys
):
    rs = metric()
    result = rs(
        load_mnist_model,
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
        **params,
    )
    with capsys.disabled():
        print(f"result = {result}")
    assert (result != np.nan).all(), "Probably divided by 0"
    assert (
        result.shape[0] == load_mnist_images["x_batch"].shape[0]
    ), "Must have same batch size"
"""
