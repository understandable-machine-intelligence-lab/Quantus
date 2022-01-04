import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "perturb_std": 0.1,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_local_lipschitz_estimate(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images,
    load_mnist_model,
):
    model = load_mnist_model
    x_batch, y_batch = (
        load_mnist_images["x_batch"].numpy(),
        load_mnist_images["y_batch"].numpy(),
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
    scores = LocalLipschitzEstimate(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_max_sensitivity(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images,
    load_mnist_model,
):
    model = load_mnist_model
    x_batch, y_batch = (
        load_mnist_images["x_batch"].numpy(),
        load_mnist_images["y_batch"].numpy(),
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
    scores = MaxSensitivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(
            ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "perturb_radius": 0.2,
                "nr_samples": 10,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_avg_sensitivity(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images,
    load_mnist_model,
):
    model = load_mnist_model
    x_batch, y_batch = (
        load_mnist_images["x_batch"].numpy(),
        load_mnist_images["y_batch"].numpy(),
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
    scores = AvgSensitivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(
            ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.fixing
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "nr_steps": 10,
                "patch_size": 7,
                "img_size": 28,
                "nr_channels": 1,
                "explain_func": explain,
                "method": "Saliency",
                "disable_warnings": True,
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_continuity(
    params: dict,
    expected: Union[float, dict, bool],
    load_mnist_images,
    load_mnist_model,
):
    model = load_mnist_model
    x_batch, y_batch = (
        load_mnist_images["x_batch"].numpy(),
        load_mnist_images["y_batch"].numpy(),
    )
    explain = params["explain_func"]
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params,
    )
    scores = Continuity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    assert scores is not None, "Test failed."
