from pytest_lazyfixture import lazy_fixture  # noqa

from ..fixtures import *
from ... import quantus
import numpy as np

"""
Following scenarios are to be tested for each Relative Stability metric

    - RS objective: 
        - single data point
            - 1 channel
            - 3 channels
        - batched
            - 1 channel
            - 3 channels 
        - vectorized over perturbations
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
    
    
We do not test, do not assert:
    - channel order of inputs/model -> it's users responsibility to provide inputs and model, which are compatible
    - perturb_func, explain_func -> if they're not callable, anyway Error will be raised when calling them
    - Meaningful values for num_perturbations, if perturbations caused enough changes  -> the library is ment for scientific usage, we assume domain experts know what they're doing
    

There are no desired values, the tests are rather just to make sure no exceptions occur during intended usage 🤷🤷  

Since all 3 relative stabilities are exactly the same, except for arguments provided to objective, 
it's enough just to test 1 class extensively
"""


@pytest.mark.robustness
@pytest.mark.parametrize(
    "x",
    [(np.random.random((32, 32, 1))), (np.random.random((32, 32, 3)))],
    ids=["1 channel", "3 channels"],
)
def test_relative_input_stability_objective_single_point(x, capsys):
    result = quantus.relative_stability_objective_single_point(
        x, x, x, x, 0.00001, True
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result != np.nan, "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "x",
    [(np.random.random((10, 32, 32, 1))), (np.random.random((10, 32, 32, 3)))],
    ids=["1 channel", "3 channels"],
)
def test_relative_input_stability_objective_batched(x, capsys):
    result = quantus.relative_stability_objective_batched(x, x, x, x, 0.00001, True)
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (10,), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "x,xs",
    [
        (np.random.random((10, 32, 32, 1)), np.random.random((5, 10, 32, 32, 1))),
        (np.random.random((10, 32, 32, 3)), np.random.random((5, 10, 32, 32, 1))),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_input_stability_objective_vectorized_over_perturbations(
        x, xs, capsys
):
    result = quantus.relative_stability_objective_vectorized_over_perturbation_axis(
        x, xs, x, xs, 0.00001, True
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (5, 10), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "h,a",
    [
        (np.random.random(10), np.random.random((32, 32, 1))),
        (np.random.random(10), np.random.random((32, 32, 3))),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_output_stability_objective_single_point(h, a, capsys):
    result = quantus.relative_stability_objective_single_point(
        h, h, a, a, 0.00001, False
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result != np.nan, "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "h,a",
    [
        (np.random.random((5, 10)), np.random.random((5, 32, 32, 1))),
        (np.random.random((5, 10)), np.random.random((5, 32, 32, 3))),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_output_stability_objective_batched(h, a, capsys):
    result = quantus.relative_stability_objective_batched(h, h, a, a, 0.00001, False)
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (5,), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "h,hs,a,a_s",
    [
        (
                np.random.random((5, 10)),
                np.random.random((10, 5, 10)),
                np.random.random((5, 32, 32, 1)),
                np.random.random((10, 5, 32, 32, 1)),
        ),
        (
                np.random.random((5, 10)),
                np.random.random((10, 5, 10)),
                np.random.random((5, 32, 32, 3)),
                np.random.random((10, 5, 32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_output_stability_objective_vectorized_over_perturbations(
        h, hs, a, a_s, capsys
):
    result = quantus.relative_stability_objective_vectorized_over_perturbation_axis(
        h, hs, a, a_s, 0.00001, False
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (
        10,
        5,
    ), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "lx,a",
    [
        (
                np.random.random(128),
                np.random.random((32, 32, 1)),
        ),
        (
                np.random.random(128),
                np.random.random((32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_representation_stability_objective_single_point(lx, a, capsys):
    result = quantus.relative_stability_objective_single_point(
        lx, lx, a, a, 0.00001, True
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result != np.nan, "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "lx,a",
    [
        (
                np.random.random((5, 128)),
                np.random.random((5, 32, 32, 1)),
        ),
        (
                np.random.random((5, 128)),
                np.random.random((5, 32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_representation_stability_objective_batched(lx, a, capsys):
    result = quantus.relative_stability_objective_batched(lx, lx, a, a, 0.00001, True)
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (5,), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "lx,lxs,a,a_s",
    [
        (
                np.random.random((5, 128)),
                np.random.random((10, 5, 128)),
                np.random.random((5, 32, 32, 1)),
                np.random.random((10, 5, 32, 32, 1)),
        ),
        (
                np.random.random((5, 128)),
                np.random.random((10, 5, 128)),
                np.random.random((5, 32, 32, 3)),
                np.random.random((10, 5, 32, 32, 3)),
        ),
    ],
    ids=["1 channel", "3 channels"],
)
def test_relative_representation_stability_objective_vectorized_over_perturbations(
        lx, lxs, a, a_s, capsys
):
    result = quantus.relative_stability_objective_vectorized_over_perturbation_axis(
        lx, lxs, a, a_s, 0.00001, True
    )
    with capsys.disabled():
        print(f"{result = }")

    assert result.shape == (
        10,
        5,
    ), "Must output same dimension as inputs batch axis"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # no explain func, no pre computed explanations
                {},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # pre-computed perturbations don't have extra batch dimension
                {"xs_batch": np.random.random((124, 28, 28, 1))},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # only a_batch given
                {"a_batch": np.random.random((124, 28, 28, 1))},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # only as_batch given
                {"as_batch": np.random.random((5, 124, 28, 28, 1))},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # pre-computed perturbed explanations have no extra batch axis
                {
                    "a_batch": np.random.random((124, 28, 28, 1)),
                    "as_batch": np.random.random((124, 28, 28, 1)),
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                # provided pre-computed perturbed explanations, but not perturbed x
                {
                    "a_batch": np.random.random((124, 28, 28, 1)),
                    "as_batch": np.random.random((5, 124, 28, 28, 1)),
                },
        ),
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
def test_invalid_kwargs(model, data, params):
    with pytest.raises(ValueError):
        ris = quantus.RelativeInputStability(**params)
        ris(model, data["x_batch"], data["y_batch"], **params)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                },
        )
    ],
)
def test_pre_computed_perturbations(model, data, params, capsys):
    ris = quantus.RelativeInputStability(**params)
    x = data["x_batch"]
    xs = np.asarray([quantus.random_noise(x) for _ in range(5)])

    result = ris(model, x, data["y_batch"], xs_batch=xs, **params)
    with capsys.disabled():
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "num_perturbations": 10,
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "num_perturbations": 10,
                    "perturb_func": quantus.gaussian_noise,
                    "indices": list(range(124)),
                    "indexed_axes": [0],
                    "perturb_std": 0.5,
                    "perturb_mean": 0.3,
                },
        ),
    ],
    ids=[
        "default perturb_func",
        "perturb_func = quantus.gaussian_noise, with extra kwargs",
    ],
)
def test_compute_perturbations(model, data, params, capsys):
    ris = quantus.RelativeInputStability(**params)
    x = data["x_batch"]

    result = ris(model, x, data["y_batch"], **params)
    with capsys.disabled():
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [(lazy_fixture("load_mnist_model_tf"), lazy_fixture("load_mnist_images_tf"), {})],
)
def test_precomputed_explanations(model, data, params, capsys):
    x = data["x_batch"]
    ex = quantus.explain(model, x, data["y_batch"])

    ris = quantus.RelativeInputStability(**params)
    result = ris(
        model,
        x,
        data["y_batch"],
        xs_batch=np.stack([x, x]),
        a_batch=ex,
        as_batch=np.stack([ex, ex]),
    )

    with capsys.disabled():
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


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
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"explain_func": quantus.explain, "num_perturbations": 10, "abs": True},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "num_perturbations": 10,
                    "normalise": True,
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "num_perturbations": 10,
                    "display_progressbar": True,
                },
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "num_perturbations": 10,
                    "return_aggregate": True,
                },
        ),
    ],
    ids=[
        "abs = True",
        "normalise = True",
        "display_progressbar = True",
        "return_aggregate = True",
    ],
)
def test_params_to_base_class(model, data, params, capsys):
    ris = quantus.RelativeInputStability(**params)

    result = ris(model, data["x_batch"], data["y_batch"], **params)
    with capsys.disabled():
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


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
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"


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
                {
                    "explain_func": quantus.explain,
                    "layer_names": ["test_conv"]
                },
        ),
        (
                lazy_fixture("load_cnn_2d_1channel_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "explain_func": quantus.explain,
                    "layer_indices": [7, 8]
                },
        ),
    ],
    ids=["leNet + mnist", "2d CNN + mnist", "conv layer only", "last 2 layers"],
)
def test_relative_representation_stability(model, data, params, capsys):
    rrs = quantus.RelativeRepresentationStability(**params)

    result = rrs(model, data["x_batch"], data["y_batch"], **params)
    with capsys.disabled():
        print(f"{result = }")

    assert (result != np.nan).all(), "Probably divided by 0"