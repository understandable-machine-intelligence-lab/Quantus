from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import (
    batch_baseline_replacement_by_indices,
    baseline_replacement_by_indices,
    noisy_linear_imputation,
)
from quantus.functions.similarity_func import (
    correlation_spearman,
    correlation_kendall_tau,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.faithfulness import (
    FaithfulnessCorrelation,
    FaithfulnessEstimate,
    Infidelity,
    IROF,
    Monotonicity,
    MonotonicityCorrelation,
    PixelFlipping,
    RegionPerturbation,
    ROAD,
    Selectivity,
    SensitivityN,
    Sufficiency,
    SymmetricRelevanceGain,
)


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
                    "perturb_baseline": "mean",
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
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
                    "perturb_baseline": "mean",
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
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
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
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
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
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
                    "perturb_baseline": "mean",
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientsInput",
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
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "subset_size": 100,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "subset_size": 10,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "subset_size": 100,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "nr_runs": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "subset_size": 784,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"exception": ValueError},
        ),
    ],
)
def test_faithfulness_correlation(
    model,
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

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = FaithfulnessCorrelation(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )[0]
        return

    scores = FaithfulnessCorrelation(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )[0]

    assert np.all(
        ((scores >= expected["min"]) & (scores <= expected["max"]))
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "uniform",
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
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 196,
                    "perturb_baseline": "uniform",
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
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "uniform",
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "uniform",
                    "abs": True,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "uniform",
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
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "uniform",
                    "features_in_step": 10,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_faithfulness_estimate(
    model,
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
    scores = FaithfulnessEstimate(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "mean",
                    "segmentation_method": "slic",
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
            {"min": 0.0, "max": 80.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "segmentation_method": "slic",
                    "normalise": True,
                    "abs": True,
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
            {"min": 0.0, "max": 80.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "mean",
                    "segmentation_method": "slic",
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
            {"min": 0.0, "max": 80.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "segmentation_method": "slic",
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"exception": ValueError},
        ),
    ],
)
def test_iterative_removal_of_features(
    model,
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

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = IROF(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )
        return

    scores = IROF(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "black",
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
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "white",
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
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "mean",
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "features_in_step": 28,
                    "perturb_baseline": "black",
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
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "black",
                    "features_in_step": 10,
                    "normalise": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"allowed_dtypes": [True, False]},
        ),
    ],
)
def test_monotonicity_arya(
    model,
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
    scores = Monotonicity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(s in expected["allowed_dtypes"] for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "eps": 1e-5,
                    "nr_samples": 10,
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
                    "perturb_baseline": "uniform",
                    "similarity_func": correlation_kendall_tau,
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
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "eps": 1e-5,
                    "nr_samples": 10,
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
                    "perturb_baseline": "uniform",
                    "similarity_func": correlation_kendall_tau,
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
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "eps": 1e-5,
                    "nr_samples": 10,
                    "features_in_step": 10,
                    "normalise": True,
                    "abs": True,
                    "perturb_baseline": "uniform",
                    "similarity_func": correlation_kendall_tau,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {},
            },
            1.0,
        ),
    ],
)
def test_monotonicity_correlation(
    model,
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
    scores = MonotonicityCorrelation(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert scores is not None, "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
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
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 14,
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
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "uniform",
                    "features_in_step": 56,
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
                    "perturb_baseline": "uniform",
                    "features_in_step": 112,
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
                "init": {
                    "perturb_baseline": "mean",
                    "features_in_step": 28,
                    "normalise": True,
                    "abs": True,
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
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "features_in_step": 10,
                    "normalise": False,
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                "call": {},
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "uniform",
                    "features_in_step": 56,
                    "normalise": True,
                    "abs": True,
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
            {"min": 0.0, "max": 14.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "features_in_step": 10,
                    "normalise": False,
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                "call": {},
            },
            {"min": 0.0, "max": 10.0},
        ),
    ],
)
def test_pixel_flipping(
    model,
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

    metric = PixelFlipping(**init_params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(
        [
            (s >= expected["min"] and s <= expected["max"])
            for s_list in scores
            for s in s_list
        ]
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "order": "morf",
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
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "order": "random",
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
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "order": "morf",
                    "disable_warnings": True,
                    "perturb_func": batch_baseline_replacement_by_indices,
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
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "order": "morf",
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
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "order": "morf",
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
    ],
)
def test_region_perturbation(
    model,
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

    metric = RegionPerturbation(**init_params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(
        [
            (s >= expected["min"] and s <= expected["max"])
            for s_list in scores
            for s in s_list
        ]
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "abs": True,
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
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "uniform",
                    "patch_size": 4,
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
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "init": {
                    "perturb_baseline": "uniform",
                    "patch_size": 4,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "VanillaGradients",
                    },
                },
            },
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "abs": True,
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
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "return_auc": True,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {"type": np.float64},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "mean",
                    "patch_size": 7,
                    "normalise": True,
                    "abs": True,
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
            {"type": np.float64},
        ),
    ],
)
def test_selectivity(
    model,
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

    metric = Selectivity(**init_params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert type(metric.get_auc_score) == expected["type"], "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.9,
                    "features_in_step": 28,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
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
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.8,
                    "features_in_step": 28,
                    "similarity_func": correlation_spearman,
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
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.7,
                    "features_in_step": 28,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
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
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.9,
                    "features_in_step": 28,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
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
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.9,
                    "features_in_step": 10,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_baseline": "black",
                    "n_max_percentage": 0.9,
                    "features_in_step": 28,
                    "similarity_func": correlation_spearman,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": True,
                    "return_aggregate": True,
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
    ],
)
def test_sensitivity_n(
    model,
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
    scores = SensitivityN(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "return_aggregate": False,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "n_perturb_samples": 10,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": False,
                "init": {
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "return_aggregate": False,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "n_perturb_samples": 5,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            {},
        ),
        # (
        #   lazy_fixture("load_cifar10_model"),
        #   lazy_fixture("load_cifar10_images"),
        #   {
        #       "a_batch_generate": True,
        #       "init": {
        #           "perturb_func": baseline_replacement_by_indices,
        #           "return_aggregate": True,
        #           "normalise": False,
        #           "abs": True,
        #           "disable_warnings": False,
        #           "display_progressbar": False,
        #           "n_perturb_samples": 5,
        #       },
        #       "call": {
        #           "explain_func": explain,
        #           "explain_func_kwargs": {
        #               "method": "Saliency",
        #           },
        #       },
        #   },
        #   {},
        # ),
    ],
)
def test_infidelity(
    model,
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

    scores = Infidelity(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert scores is not None, "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "perturb_func": noisy_linear_imputation,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "percentages": list(range(1, 100, 2)),
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
                    "perturb_func": noisy_linear_imputation,
                    "normalise": True,
                    "abs": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "percentages": list(range(1, 100, 2)),
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
def test_ROAD(
    model,
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
    scores = ROAD(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert all(s <= expected["max"] for s in scores.values()) & (
        all(s >= expected["min"] for s in scores.values())
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "init": {
                    "threshold": 0.2,
                    "normalise": False,
                    "abs": False,
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
                    "threshold": 0.6,
                    "normalise": True,
                    "abs": True,
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
    ],
)
def test_sufficiency(
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

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = Sufficiency(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )[0]
        return

    scores = Sufficiency(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )[0]
    assert (scores >= expected["min"]) & (scores <= expected["max"]), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "a_batch_generate": True,
                "init": {
                    "features_in_step": 28,
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
                "a_batch_generate": True,
                "init": {
                    "features_in_step": 28,
                    "perturb_baseline": "black",
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
                "a_batch_generate": True,
                "init": {
                    "features_in_step": 56,
                    "perturb_func": batch_baseline_replacement_by_indices,
                    "perturb_func_kwargs": {},
                    "perturb_baseline": "mean",
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
                "a_batch_generate": True,
                "init": {
                    "features_in_step": 28,
                    "normalise": True,
                    "return_aggregate": True,
                    "aggregate_func": np.mean,
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
            {"min": -1.0, "max": 1.0, "n_scores": 1},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "a_batch_generate": False,
                "init": {
                    "features_in_step": 10,
                    "normalise": False,
                    "perturb_baseline": "mean",
                    "disable_warnings": True,
                },
                "call": {},
            },
            {"min": -1.0, "max": 1.0},
        ),
    ],
)
def test_symmetric_relevance_gain(
    model,
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
        explain_func = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain_func(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    metric = SymmetricRelevanceGain(**init_params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )

    assert len(scores) == expected.get("n_scores", len(x_batch)), "Test failed."
    assert all(np.isfinite(s) for s in scores), "Test failed."
    assert all(
        (s >= expected["min"] and s <= expected["max"]) for s in scores
    ), "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_sign_flip(load_mnist_model, load_mnist_images):
    """Negating the attributions swaps the MIF and LIF orderings, so the score flips sign."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]
    a_batch = np.random.randn(*x_batch.shape)

    metric = SymmetricRelevanceGain(
        features_in_step=28, normalise=False, abs=False, disable_warnings=True
    )
    scores = metric(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )
    scores_neg = metric(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=-a_batch
    )

    assert np.allclose(scores, -np.asarray(scores_neg), atol=1e-6), "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_endpoints(
    load_mnist_model, load_mnist_images, monkeypatch
):
    """Both curves share the unoccluded and the fully occluded points."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]

    mif_curves_batches, lif_curves_batches = [], []
    compute_curves = SymmetricRelevanceGain._compute_curves_torch

    def spy(self, *args, **kwargs):
        curves_mif, curves_lif = compute_curves(self, *args, **kwargs)
        mif_curves_batches.append(curves_mif)
        lif_curves_batches.append(curves_lif)
        return curves_mif, curves_lif

    monkeypatch.setattr(SymmetricRelevanceGain, "_compute_curves_torch", spy)

    metric = SymmetricRelevanceGain(features_in_step=28, disable_warnings=True)
    metric(
        model=load_mnist_model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        explain_func=explain,
        explain_func_kwargs={"method": "Saliency"},
    )

    mif_curves = np.concatenate(mif_curves_batches, axis=0)
    lif_curves = np.concatenate(lif_curves_batches, axis=0)
    assert mif_curves.shape == (len(x_batch), 29), "Test failed."
    assert lif_curves.shape == (len(x_batch), 29), "Test failed."
    assert np.allclose(mif_curves[:, 0], lif_curves[:, 0]), "Test failed."
    assert np.allclose(mif_curves[:, -1], lif_curves[:, -1]), "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_torch_path_equals_numpy_path(
    load_mnist_model, load_mnist_images, monkeypatch
):
    """The torch-resident fast path and the generic numpy path agree."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]
    a_batch = np.random.randn(*x_batch.shape)

    metric = SymmetricRelevanceGain(
        features_in_step=28, normalise=False, disable_warnings=True
    )
    scores_torch = metric(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )

    monkeypatch.setattr(
        SymmetricRelevanceGain, "_can_use_torch_fast_path", lambda self, model: False
    )
    scores_numpy = metric(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )

    assert np.allclose(scores_torch, scores_numpy, atol=1e-5), "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_random_attribution(
    load_mnist_model, load_mnist_images
):
    """Random attributions score approximately zero on average."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]
    a_batch = np.random.randn(*x_batch.shape)

    metric = SymmetricRelevanceGain(
        features_in_step=28, normalise=False, disable_warnings=True
    )
    scores = metric(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )

    assert np.abs(np.mean(scores)) < 0.1, "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_explicit_default_perturb_func(
    load_mnist_model, load_mnist_images
):
    """Passing the default perturb_func explicitly behaves like perturb_func=None."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]
    a_batch = np.random.randn(*x_batch.shape)

    metric_default = SymmetricRelevanceGain(
        features_in_step=28, normalise=False, disable_warnings=True
    )
    metric_explicit = SymmetricRelevanceGain(
        features_in_step=28,
        perturb_func=batch_baseline_replacement_by_indices,
        perturb_func_kwargs={},
        normalise=False,
        disable_warnings=True,
    )

    scores_default = metric_default(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )
    scores_explicit = metric_explicit(
        model=load_mnist_model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
    )

    assert np.allclose(scores_default, scores_explicit, atol=1e-6), "Test failed."


@pytest.mark.faithfulness
def test_symmetric_relevance_gain_invalid_features_in_step(
    load_mnist_model, load_mnist_images
):
    """An explicit features_in_step must divide the flattened feature count."""
    x_batch, y_batch = load_mnist_images["x_batch"], load_mnist_images["y_batch"]

    metric = SymmetricRelevanceGain(features_in_step=53, disable_warnings=True)
    with pytest.raises(AssertionError):
        metric(
            model=load_mnist_model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=None,
            explain_func=explain,
            explain_func_kwargs={"method": "Saliency"},
        )
