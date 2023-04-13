from typing import Union, Dict
from pytest_lazyfixture import lazy_fixture
import pytest
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.functions.discretise_func import floating_points, rank, sign, top_n_sign
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.robustness import (
    AvgSensitivity,
    Consistency,
    Continuity,
    LocalLipschitzEstimate,
    MaxSensitivity,
)
from quantus.functions.perturb_func import synonym_replacement, spelling_replacement

# ----------------------- sensitivity -----------------------

sensitivity_tests = pytest.mark.parametrize(
    "model,data,params",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "init": {},
                "call": {
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
        ),
        pytest.param(
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {"return_aggregate": True},
                "call": {
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            id="torch_mnist",
        ),
        pytest.param(
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "abs": True,
                    "normalise": True,
                },
                "call": {
                    "explain_func_kwargs": {
                        "method": "VanillaGradients",
                    },
                },
            },
            id="tf_mnist",
        ),
        # ------------ NLP -------------
        # pytest.param(
        #    lazy_fixture("tf_sst2_model"),
        #    lazy_fixture("sst2_dataset"),
        #    {
        #        "a_batch_generate": False,
        #        "init": {"perturb_func": synonym_replacement},
        #        "call": {},
        #    },
        #    marks=[pytest.mark.nlp],
        #    id="tf_nlp_plain_text",
        # ),
        # pytest.param(
        #    lazy_fixture("torch_sst2_model"),
        #    lazy_fixture("sst2_dataset"),
        #    {
        #        "a_batch_generate": False,
        #        "init": {
        #            "perturb_func": spelling_replacement,
        #        },
        #        "call": {},
        #    },
        #    marks=[pytest.mark.nlp],
        #    id="torch_nlp_plain_text",
        # ),
        pytest.param(
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "a_batch_generate": False,
                "init": {},
                "call": {},
            },
            marks=[pytest.mark.nlp],
            id="tf_nlp_latent",
        ),
        # pytest.param(
        #    lazy_fixture("torch_sst2_model"),
        #    lazy_fixture("sst2_dataset"),
        #    {
        #        "a_batch_generate": False,
        #        "init": {},
        #        "call": {},
        #    },
        #    marks=[pytest.mark.nlp],
        #    id="torch_nlp_latent",
        # ),
    ],
)


@pytest.mark.robustness
@sensitivity_tests
def test_max_sensitivity(model, data, params, sst2_tokenizer):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = MaxSensitivity(
        **init_params, lower_bound=0.2, nr_samples=10, disable_warnings=True
    )(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        explain_func=explain,
        tokenizer=sst2_tokenizer,
        **call_params,
    )

    if init_params.get("return_aggregate", False):
        assert scores.shape == ()
    else:
        assert isinstance(scores, np.ndarray)

    assert np.all(np.asarray(scores) >= 0)
    # assert np.all(np.asarray(scores) <= 1.)


@pytest.mark.robustness
@sensitivity_tests
def test_avg_sensitivity(model, data, params, sst2_tokenizer):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = AvgSensitivity(
        **init_params, lower_bound=0.2, nr_samples=10, disable_warnings=True
    )(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        explain_func=explain,
        tokenizer=sst2_tokenizer,
        **call_params,
    )
    if init_params.get("return_aggregate", False):
        assert scores.shape == ()
    else:
        assert isinstance(scores, np.ndarray)

    assert np.all(np.asarray(scores) >= 0)
    # assert np.all(np.asarray(scores) <= 1.)


# --------------------------------------------------------------------------
@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
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
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
                    "perturb_std": 0.1,
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
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
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
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
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
    ],
)
def test_local_lipschitz_estimate(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
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
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = LocalLipschitzEstimate(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
                    "patch_size": 10,
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
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
                    "patch_size": 7,
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
                    "nr_steps": 10,
                    "patch_size": 10,
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
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
                    "patch_size": 7,
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
                "a_batch_generate": False,
                "init": {
                    "nr_steps": 10,
                    "patch_size": 10,
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
            {"exception": ValueError},
        ),
    ],
)
def test_continuity(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
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
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if isinstance(expected, Dict):
        if "exception" in expected:
            with pytest.raises(expected["exception"]):
                Continuity(**init_params)(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    **call_params,
                )
            return

    scores = Continuity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )
    if isinstance(expected, float):
        assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": floating_points,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": sign,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": top_n_sign,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "discretise_func": rank,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_consistency(
    model: ModelInterface,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **call_params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = Consistency(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )
        return

    scores = Consistency(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )[0]
    assert (scores >= expected["min"]) & (scores <= expected["max"]), "Test failed."


# -------------------return_nan_when_prediction_changes=True-----------------------


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric,model,data, init_kwargs, call_kwargs",
    [
        (
            LocalLipschitzEstimate,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        (
            LocalLipschitzEstimate,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        (
            AvgSensitivity,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        (
            AvgSensitivity,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        (
            MaxSensitivity,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        (
            MaxSensitivity,
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {},
            {"explain_func_kwargs": {"method": "Saliency"}},
        ),
        # ------------- NLP -----------
        # Requires to mock __call__, which will cause gradients to be None
        # pytest.param(
        #     AvgSensitivity,
        #     lazy_fixture("tf_sst2_model"),
        #     lazy_fixture("sst2_dataset"),
        #     {},
        #     {"explain_func_kwargs": {"method": "GradNorm"}},
        #     marks=pytest.mark.nlp,
        # ),
        pytest.param(
            MaxSensitivity,
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"perturb_func": spelling_replacement},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            marks=pytest.mark.nlp,
        ),
    ],
    ids=[
        "lipschitz_mnist",
        "lipschitz_tabular",
        "avg_sen_mnist",
        "avg_sen_tabular",
        "max_sen_mnist",
        "max_sen_tabular",
        # "avg_sen_NLP",
        "max_sen_NLP_plain_text",
    ],
)
def test_return_nan_when_prediction_changes(
    metric, model, data, init_kwargs, call_kwargs, mock_prediction_changed
):
    # This test case requires different set-up and assertions, so we have it in separate function.
    metric_instance = metric(
        disable_warnings=True,
        nr_samples=10,
        return_nan_when_prediction_changes=True,
        **init_kwargs,
    )
    result = metric_instance(
        model,
        data["x_batch"],
        data["y_batch"],
        explain_func=explain,
        **call_kwargs,
    )
    assert np.isnan(result).all()


@pytest.mark.robustness
def test_return_nan_when_prediction_changes_continuity(
    load_mnist_model, load_mnist_images, mock_prediction_changed
):
    # Continuity returns dict, so we have it in separate function in order to keep assertions readable.
    metric_instance = Continuity(
        disable_warnings=True,
        return_nan_when_prediction_changes=True,
        nr_steps=10,
        patch_size=7,
    )
    result = metric_instance(
        load_mnist_model,
        load_mnist_images["x_batch"],
        load_mnist_images["y_batch"],
        explain_func=explain,
        explain_func_kwargs={
            "method": "Saliency",
        },
    )
    for i in result:
        values = list(i.values())
        # Last element of scores is output logits, obviously they're not nan.
        for v in values[:-1]:
            assert np.isnan(v).any()
