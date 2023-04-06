import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.explanation_func import explain
from quantus.functions.similarity_func import correlation_spearman, correlation_pearson
from quantus.metrics.randomisation import ModelParameterRandomisation, RandomLogit


def explain_func_stub(model, inputs, targets, **kwargs):
    # tf-explain does not support 2D inputs
    return np.random.default_rng(42).uniform(low=0, high=0.5, size=inputs.shape)


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "layer_order": "top_down",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "layer_order": "top_down",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "layer_order": "bottom_up",
                    "similarity_func": correlation_pearson,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "layer_order": "bottom_up",
                    "similarity_func": correlation_pearson,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        pytest.param(
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "layer_order": "top_down",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {"method": "VanillaGradients",},
                },
            },
            {"min": -1.0, "max": 1.0},
            marks=pytest.mark.xfail,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model_tf"),
            lazy_fixture("almost_uniform_1d_no_abatch_channel_last"),
            {
                "a_batch_generate": False,
                "init": {
                    "layer_order": "bottom_up",
                    "similarity_func": correlation_pearson,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {"method": "VanillaGradients",},
                },
            },
            {"exception": ValueError},
        ),
        pytest.param(
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "a_batch_generate": False,
                "init": {
                    "layer_order": "bottom_up",
                    "similarity_func": correlation_pearson,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {"method": "Gradient",},
                },
            },
            {"min": -1.0, "max": 1.0},
            # marks=pytest.mark.xfail,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "layer_order": "top_down",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "layer_order": "independent",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "layer_order": "independent",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "reduce_axes": (),
                    },
                },
            },
            {"min": -1.0, "max": 1.01},
        ),
        (
            lazy_fixture("titanic_model_tf"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "layer_order": "independent",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain_func_stub},
            },
            {"min": -1.0, "max": 1.0},
        ),
        pytest.param(
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "a_batch_generate": False,
                "init": {
                    "layer_order": "independent",
                    "similarity_func": correlation_spearman,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain},
            },
            {"min": -1.0, "max": 1.01},
            marks=pytest.mark.nlp,
        ),
        pytest.param(
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "a_batch_generate": False,
                "init": {
                    "layer_order": "independent",
                    "similarity_func": correlation_spearman,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain},
            },
            {"min": -1.0, "max": 1.01},
            marks=pytest.mark.nlp,
        ),
    ],
)
def test_model_parameter_randomisation(
    model,
    data,
    params,
    expected,
    torch_device
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            device=torch_device,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores_layers = ModelParameterRandomisation(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
                device=torch_device
            )
        return

    scores_layers = ModelParameterRandomisation(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
        device=torch_device
    )
    if isinstance(expected, float):
        assert all(
            s == expected for layer, scores in scores_layers.items() for s in scores
        ), "Test failed."
    else:
        assert all(
            ((s >= expected["min"]) & (s <= expected["max"]))
            for layer, scores in scores_layers.items()
            for s in scores
        ), "Test failed."


@pytest.mark.randomisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "num_classes": 10,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "num_classes": 10,
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "num_classes": 10,
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {
                    "num_classes": 10,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "num_classes": 10,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("titanic_model_torch"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "num_classes": 2,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                        "reduce_axes": (),
                    },
                },
            },
            {"min": -1.0, "max": 1.01},
        ),
        (
            lazy_fixture("titanic_model_tf"),
            lazy_fixture("titanic_dataset"),
            {
                "init": {
                    "num_classes": 2,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain_func_stub},
            },
            {"min": -1.0, "max": 1.01},
        ),
        pytest.param(
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "a_batch_generate": False,
                "init": {
                    "num_classes": 2,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain},
            },
            {"min": -1.0, "max": 1.01},
            marks=pytest.mark.nlp,
        ),
        pytest.param(
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "a_batch_generate": False,
                "init": {
                    "num_classes": 2,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": True,
                },
                "call": {"explain_func": explain},
            },
            {"min": -1.0, "max": 1.01},
            marks=pytest.mark.nlp,
        ),
    ],
)
def test_random_logit(
    model,
    data,
    params,
    expected,
    torch_device
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            device=torch_device,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = RandomLogit(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
        device=torch_device
    )

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."
