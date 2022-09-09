import pytest
from pytest_lazyfixture import lazy_fixture  # noqa
import numpy as np

from ..fixtures import *  # noqa
from ... import quantus

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
        (np.random.random((10, 32, 32, 3)), np.random.random((10, 32, 32, 1))),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_input_stability_objective(x, xs, capsys):
    result = quantus.RelativeInputStability.relative_input_stability_objective(x, xs, x, xs)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert result.shape[0] == x.shape[0], "Must output same dimension as inputs batch axis"


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
    result = quantus.RelativeOutputStability.relative_output_stability_objective(h, hs, a, a_s)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert result.shape[0] == h.shape[0], "Must output same dimension as inputs batch axis"


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
    result = quantus.RelativeRepresentationStability.relative_representation_stability_objective(lx, lxs, a, a_s)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Nans are not allowed"
    assert result.shape[0] == lx.shape[0], "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params",
    [
        # no explain func, no pre computed explanations
        {},
        # pre-computed perturbations don't have extra batch dimension
        {"xs_batch": np.random.random((124, 28, 28, 1))},
        # only a_batch given
        {"a_batch": np.random.random((124, 28, 28, 1))},
        # only as_batch given
        {"as_batch": np.random.random((5, 124, 28, 28, 1))},
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
def test_invalid_kwargs(load_mnist_model_tf, load_mnist_images_tf, params):
    with pytest.raises(ValueError):
        ris = quantus.RelativeInputStability(**params)
        ris(load_mnist_model_tf, load_mnist_images_tf["x_batch"], load_mnist_images_tf["y_batch"], **params)


@pytest.mark.robustness
def test_pre_computed_perturbations(load_mnist_images_tf, load_mnist_model_tf, capsys):
    ris = quantus.RelativeInputStability()
    x = load_mnist_images_tf["x_batch"]
    xs = np.asarray([quantus.random_noise(x) for _ in range(5)])

    result = ris(
        load_mnist_model_tf,
        x,
        load_mnist_images_tf["y_batch"],
        xs_batch=xs,
        explain_func=quantus.explain
    )
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == x.shape[0], "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params",
    [
        {
            "explain_func": quantus.explain,
        },
        {
            "explain_func": quantus.explain,
            "perturb_func": quantus.gaussian_noise,
            "indices": list(range(124)),
            "indexed_axes": [0],
            "perturb_std": 0.5,
            "perturb_mean": 0.3,
        },

    ],
    ids=[
        "default perturb_func",
        "perturb_func = quantus.gaussian_noise, with extra kwargs",
    ],
)
def test_compute_perturbations(load_mnist_model_tf, load_mnist_images_tf, params, capsys):
    ris = quantus.RelativeInputStability(**params)
    x = load_mnist_images_tf["x_batch"]

    result = ris(load_mnist_model_tf, x, load_mnist_images_tf["y_batch"], **params)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == x.shape[0], "Must have same batch size"


@pytest.mark.robustness
def test_precomputed_explanations(load_mnist_model_tf, load_mnist_images_tf, capsys):
    x = load_mnist_images_tf["x_batch"]
    ex = quantus.explain(load_mnist_model_tf, x, load_mnist_images_tf["y_batch"])

    ris = quantus.RelativeInputStability()
    result = ris(
        load_mnist_model_tf,
        x,
        load_mnist_images_tf["y_batch"],
        xs_batch=np.stack([x, x]),
        a_batch=ex,
        as_batch=np.stack([ex, ex]),
    )

    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == data["x_batch"].shape[0], "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "method": "Gradient",
                    "num_perturbations": 10,
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "method": "IntegratedGradients",
                    "num_perturbations": 10,
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "method": "InputXGradient",
                    "num_perturbations": 10,
                },
        ),
        (
                lazy_fixture("load_cnn_2d_3channels_tf"),
                lazy_fixture("load_cifar10_images"),
                {
                    "explain_func": quantus.explain,
                    "method": "Occlusion",
                    "num_perturbations": 100,
                },
        ),
        (
                lazy_fixture("load_cnn_2d_3channels_tf"),
                lazy_fixture("load_cifar10_images"),
                {
                    "explain_func": quantus.explain,
                    "method": "GradCam",
                    "num_perturbations": 100,
                    "gc_layer": "test_conv",
                },
        ),
    ],
    ids=[
        "method = Gradient",
        "method = IntegratedGradients",
        "method = InputXGradient",
        "method = Occlusion",
        "method = GradCam",
    ],
)
def test_compute_explanations(model, data, params, capsys):
    ris = quantus.RelativeInputStability(**params)

    result = ris(model, data["x_batch"], data["y_batch"], **params)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == data["x_batch"].shape[0], "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric, params",
    [
        (quantus.RelativeInputStability, {"abs": True}),
        (quantus.RelativeInputStability, {"normalise": True}),
        (quantus.RelativeInputStability, {"display_progressbar": True}),
        (quantus.RelativeInputStability, {"return_aggregate": True}),
        (quantus.RelativeOutputStability, {"abs": True}),
        (quantus.RelativeOutputStability, {"normalise": True}),
        (quantus.RelativeOutputStability, {"display_progressbar": True}),
        (quantus.RelativeOutputStability, {"return_aggregate": True}),
        (quantus.RelativeRepresentationStability, {"abs": True}),
        (quantus.RelativeRepresentationStability, {"normalise": True}),
        (quantus.RelativeRepresentationStability, {"display_progressbar": True}),
        (quantus.RelativeRepresentationStability, {"return_aggregate": True}),
    ],
    ids=[
        "RIS + abs = True",
        "RIS + normalise = True",
        "RIS + display_progressbar = True",
        "RIS + return_aggregate = True",
        "ROS + abs = True",
        "ROS + normalise = True",
        "ROS + display_progressbar = True",
        "ROS + return_aggregate = True",
        "RRS + abs = True",
        "RRS + normalise = True",
        "RRS + display_progressbar = True",
        "RRS + return_aggregate = True",
    ],
)
def test_params_to_base_class(metric, params):
    ris = metric(**params)
    for i in params:
        attr = getattr(ris, i)
        assert attr == params[i], "Parameter was not initialized"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain, "num_perturbations": 10},
        ),
    ],
)
def test_relative_output_stability(model, data, params, capsys):
    ros = quantus.RelativeOutputStability(**params)

    result = ros(model, data["x_batch"], data["y_batch"], **params)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == data["x_batch"].shape[0], "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain},
        ),
        (
                # This situation caused problems in tutorials
                lazy_fixture("load_cnn_2d_1channel_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain},
        ),
        (
                lazy_fixture("load_cnn_2d_1channel_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain, "layer_names": ["test_conv"]},
        ),
        (
                lazy_fixture("load_cnn_2d_1channel_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain, "layer_indices": [7, 8]},
        ),
    ],
    ids=["leNet + mnist", "2d CNN + mnist", "conv layer only", "last 2 layers"],
)
def test_relative_representation_stability(model, data, params, capsys):
    rrs = quantus.RelativeRepresentationStability(**params)

    result = rrs(model, data["x_batch"], data["y_batch"], **params)
    with capsys.disabled():
        print(f"result = {result}")

    assert (result != np.nan).all(), "Probably divided by 0"
    assert result.shape[0] == data["x_batch"].shape[0], "Must have same batch size"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric, params",
    [
        (
                quantus.RelativeInputStability,
                {"explain_func": quantus.explain, "num_perturbations": 10},
        ),
        (
                quantus.RelativeOutputStability,
                {"explain_func": quantus.explain, "num_perturbations": 10},
        ),
        (
                quantus.RelativeRepresentationStability,
                {"explain_func": quantus.explain, "num_perturbations": 10},
        ),
    ],
    ids=["RIS", "ROS", "RRS"],
)
def test_relative_stability_pytorch(
        load_mnist_model, load_mnist_images, metric, params, capsys
):
    rs = metric(**params)
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
