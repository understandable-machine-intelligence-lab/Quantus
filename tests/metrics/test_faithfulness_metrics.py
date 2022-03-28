from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers import perturb_func
from ...quantus.helpers.explanation_func import explain


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "nr_runs": 10,
                "perturb_baseline": "mean",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "nr_runs": 10,
                "perturb_baseline": "mean",
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "IntegratedGradients",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "InputXGradient",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "mean",
                "explain_func": explain,
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "subset_size": 100,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "mean",
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "subset_size": 10,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "mean",
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "subset_size": 100,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "mean",
                "explain_func": explain,
                "nr_runs": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "subset_size": 784,
                "disable_warnings": False,
                "display_progressbar": False,
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = FaithfulnessCorrelation(**params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **params,
            )[0]
        return

    scores = FaithfulnessCorrelation(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "uniform",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "uniform",
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "uniform",
                "abs": True,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "uniform",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "uniform",
                "features_in_step": 10,
                "normalise": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = FaithfulnessEstimate(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "perturb_baseline": "mean",
                "segmentation_method": "slic",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "segmentation_method": "slic",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "segmentation_method": "slic",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_baseline": "mean",
                "segmentation_method": "slic",
                "normalise": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
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

    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = IterativeRemovalOfFeatures(**params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **params,
            )
        return

    scores = IterativeRemovalOfFeatures(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "black",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "white",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "mean",
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "black",
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"allowed_dtypes": [True, False]},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_func": baseline_replacement_by_indices,
                "perturb_baseline": "black",
                "features_in_step": 10,
                "normalise": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = MonotonicityArya(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "eps": 1e-5,
                "nr_samples": 10,
                "features_in_step": 28,
                "normalise": True,
                "abs": True,
                "perturb_baseline": "uniform",
                "similarity_func": correlation_kendall_tau,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "eps": 1e-5,
                "nr_samples": 10,
                "features_in_step": 28,
                "normalise": True,
                "abs": True,
                "perturb_baseline": "uniform",
                "similarity_func": correlation_kendall_tau,
                "explain_func": explain,
                "method": "Saliency",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
                "a_batch_generate": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "eps": 1e-5,
                "nr_samples": 10,
                "features_in_step": 10,
                "normalise": True,
                "abs": True,
                "perturb_baseline": "uniform",
                "similarity_func": correlation_kendall_tau,
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            1.0,
        ),
    ],
)
def test_monotonicity_nguyen(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = MonotonicityNguyen(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "perturb_baseline": "mean",
                "features_in_step": 28,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "features_in_step": 14,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "random",
                "features_in_step": 56,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "a_batch_generate": True,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "random",
                "features_in_step": 112,
                "normalise": False,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "features_in_step": 28,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "features_in_step": 10,
                "normalise": False,
                "perturb_func": perturb_func.baseline_replacement_by_indices,
                "perturb_baseline": "mean",
                "disable_warnings": True,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = PixelFlipping(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores[0]
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "order": "morf",
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": True,
            },
            {"min": -1, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "order": "random",
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "order": "morf",
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "perturb_func": perturb_func.baseline_replacement_by_patch,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "order": "morf",
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1, "max": 1.0},
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = RegionPerturbation(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
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
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "random",
                "patch_size": 4,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "perturb_baseline": "random",
                "patch_size": 4,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_baseline": "mean",
                "patch_size": 7,
                "normalise": True,
                "abs": True,
                "max_steps_per_input": 2,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": 0.0, "max": 1.0},
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
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = Selectivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores[0]
    ), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "black",
                "n_max_percentage": 0.9,
                "features_in_step": 28,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "black",
                "n_max_percentage": 0.8,
                "features_in_step": 28,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "black",
                "n_max_percentage": 0.7,
                "features_in_step": 28,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Gradient",
                "max_steps_per_input": 2,
                "disable_warnings": True,
                "display_progressbar": False,
                "a_batch_generate": False,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "perturb_baseline": "black",
                "n_max_percentage": 0.9,
                "features_in_step": 28,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "explain_func": explain,
                "method": "Saliency",
                "abs": True,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            {"min": -1.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {
                "perturb_baseline": "black",
                "n_max_percentage": 0.9,
                "features_in_step": 10,
                "similarity_func": correlation_spearman,
                "normalise": True,
                "abs": True,
                "disable_warnings": False,
                "display_progressbar": False,
                "a_batch_generate": False,
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

    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = SensitivityN(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    assert all(
        ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
    ), "Test failed."
