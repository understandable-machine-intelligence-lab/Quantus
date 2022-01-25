import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers.pytorch_model import PyTorchModel


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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "IntegratedGradients",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "InputXGradient",
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": -1.0, "max": 1.0},
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "perturb_func": baseline_replacement_by_indices,
                "features_in_step": 28,
                "perturb_baseline": "black",
                "normalise": True,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "perturb_baseline": "uniform",
                "similarity_func": correlation_kendall_tau,
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "order": "lorf",
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": -1, "max": 1.0},
        ),
    ],
)
def test_region_segmentation(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Gradient",
                "img_size": 28,
                "nr_channels": 1,
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

    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
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
                "disable_warnings": True,
                "explain_func": explain,
                "method": "Saliency",
                "img_size": 28,
                "nr_channels": 1,
            },
            {"min": 0.0, "max": 1.0},
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
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
    scores = IterativeRemovalOfFeatures(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    assert scores is not None, "Test failed."
