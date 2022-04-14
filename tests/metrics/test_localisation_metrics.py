from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain


@pytest.fixture
def all_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_2d_3ch():
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
def all_in_gt_no_abatch_1d_1ch():
    s_batch = np.zeros((10, 1, 28))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 28))
    s_batch[:, :, 0:15] = 1.0
    return {
        "x_batch": np.random.randn(10, 1, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_no_abatch_2d_1ch():
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
def all_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_zeros_2d_3ch():
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
def all_in_gt_non_normalised_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 20, size=(10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 25
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_non_normalised_2d():
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
def all_in_gt_seg_bigger_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 0:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_seg_bigger_2d_3ch():
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
def none_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_2d_3ch():
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
def none_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_zeros_2d_3ch():
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
def none_in_gt_fourth_1d():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:112] = 1.0
    a_batch[:, :, 112:224] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_fourth_2d_3ch():
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
def half_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 50:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_zeros_2d_3ch():
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
def half_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 50:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_2d_3ch():
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


# TODO: unused fixture
@pytest.fixture
def half_in_gt_zeros_bigger_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


# TODO: unused fixture
@pytest.fixture
def half_in_gt_zeros_bigger_2d_3ch():
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
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            False,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            False,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            True,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            True,
        ),
    ],
)
def test_pointing_game(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = PointingGame(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    if isinstance(expected, bool):
        assert all(s == expected for s in scores), "Test failed."
    elif isinstance(expected, dict):
        assert isinstance(scores, expected["type"]), "Test failed."
    elif isinstance(expected, list):
        assert all(s == e for s, e in zip(scores, expected)), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "k": 100,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "k": 10000,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "k": 200,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "k": 40000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.25,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "k": 20,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "k": 500,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "k": 100,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "k": 10000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "k": 200,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.38,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "k": 40000,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.1, "max": 0.25},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "k": 50,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.98,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "k": 2500,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "k": 125,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.4,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "k": 1250,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "k": 100,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "k": 10000,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "k": 100,
                "concept_influence": True,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            2.24,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "k": 10000,
                "concept_influence": True,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            5.0176,  # TODO: verify correctness
        ),
    ],
)
def test_top_k_intersection(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = TopKIntersection(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    print(scores, expected)
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_seg_bigger_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_seg_bigger_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_relevance_rank_accuracy(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = RelevanceRankAccuracy(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    if isinstance(expected, float):
        print(scores)
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_seg_bigger_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_seg_bigger_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_relevance_mass_accuracy(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = RelevanceMassAccuracy(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_non_normalised_1d_3ch"),
            {
                "normalise": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_non_normalised_2d"),
            {
                "normalise": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_fourth_1d"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_fourth_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.33333333333333337,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
    ],
)
def test_auc(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = AUC(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), f"Test failed. {scores[0]}"
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "weighted": False,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "weighted": False,
                "disable_warnings": False,
                "display_progressbar": False,
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "weighted": False,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "weighted": False,
                "explain_func": explain,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.8, "max": 0.95},  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            {"min": 0.8, "max": 0.85},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "weighted": True,
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "weighted": True,
                "abs": False,
                "disable_warnings": True,
                "display_progressbar": False,
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "weighted": False,
                "disable_warnings": True,
                "display_progressbar": True,
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
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
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    scores = AttributionLocalisation(**params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        print(scores)
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."
