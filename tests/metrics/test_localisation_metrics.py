import numpy as np
import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


@pytest.fixture
def all_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_no_abatch():
    s_batch = np.zeros((10, 1, 28, 28))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 28, 28))
    s_batch[:, :, 0:15, 0:15] = 1.0
    return {
        "x_batch": np.random.randn(10, 1, 28, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_non_normalised():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 20, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 25
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_seg_bigger():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:150, 0:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_fourth():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:112, 0:112] = 1.0
    a_batch[:, :, 112:224, 112:224] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_zeros_bigger():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("none_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            False,
        ),
        (
            lazy_fixture("half_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            True,
        ),
    ],
)
def test_pointing_game(
    data: dict, params: dict, expected: Union[bool, dict], load_mnist_model
):
    model = load_mnist_model
    scores = PointingGame(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, bool):
        assert all(s == expected for s in scores), "Test failed."
    elif isinstance(expected, dict):
        assert isinstance(scores, expected["type"]), "Test failed."
    elif isinstance(expected, list):
        assert all(s == e for s, e in zip(scores, expected)), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt"),
            {
                "k": 10000,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "k": 40000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.25,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "k": 10000,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("none_in_gt"),
            {
                "k": 10000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("none_in_gt_zeros"),
            {
                "k": 40000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.1, "max": 0.25},
        ),
        (
            lazy_fixture("half_in_gt_zeros"),
            {
                "k": 2500,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("half_in_gt_zeros"),
            {
                "k": 1250,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "k": 10000,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_top_k_intersection(
    data: dict, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    scores = TopKIntersection(**params)(
        model=load_mnist_model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("all_in_gt_seg_bigger"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("none_in_gt"),
            {
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("half_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_relevance_rank_accuracy(
    data: dict, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    scores = RelevanceRankAccuracy(**params)(
        model=load_mnist_model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt_zeros"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("all_in_gt_seg_bigger"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("none_in_gt_zeros"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("half_in_gt_zeros"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("all_in_gt_zeros"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_relevance_mass_accuracy(
    data: dict, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    scores = RelevanceMassAccuracy(**params)(
        model=load_mnist_model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("all_in_gt_non_normalised"),
            {
                "normalise": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("none_in_gt_fourth"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.33333333333333337,
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_auc(
    data: dict, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    scores = AUC(**params)(
        model=load_mnist_model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("all_in_gt_zeros"),
            {
                "weighted": False,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("all_in_gt_no_abatch"),
            {
                "weighted": False,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("all_in_gt"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.8, "max": 0.85},
        ),
        (
            lazy_fixture("none_in_gt_zeros"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("none_in_gt_zeros"),
            {
                "weighted": True,
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("all_in_gt_zeros"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_attribution_localisation(
    data: dict, params: dict, expected: Union[float, dict, bool], load_mnist_model
):
    scores = AttributionLocalisation(**params)(
        model=load_mnist_model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."
