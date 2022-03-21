from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def atts_half():
    return {"a": np.array([-1, 1, 1]), "b": np.array([0, 0, 2])}


@pytest.fixture
def atts_diff():
    return {"a": np.array([0, 1, 0, 1]), "b": np.array([1, 2, 1, 0])}


@pytest.fixture
def atts_same():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a, "b": a}


@pytest.fixture
def atts_same_linear():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a, "b": a * 3}


@pytest.fixture()
def atts_inverse():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a, "b": a * -3}


@pytest.fixture
def atts_lip_same():
    return {
        "a": np.array([-1, 1, 1]),
        "b": np.array([0, 0, 2]),
        "c": np.array([-1, 1, 1]),
        "d": np.array([0, 0, 2]),
    }


@pytest.fixture
def atts_lip_diff():
    return {
        "a": np.array([-1, 1, 1]),
        "b": np.array([0, 0, 2]),
        "c": np.array([-1, 1, 1]),
        "d": np.array([0, 0, 2]),
    }


@pytest.fixture
def atts_ssim_same():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a, "b": a}


@pytest.fixture
def atts_ssim_diff():
    return {"a": np.zeros((16, 16)), "b": np.ones((16, 16))}


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 1.0),
        (lazy_fixture("atts_same_linear"), {}, 1.0),
        (lazy_fixture("atts_diff"), {}, 0.0),
        (lazy_fixture("atts_half"), {}, 0.5),
        (lazy_fixture("atts_inverse"), {}, -1),
    ],
)
def test_correlation_spearman(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = correlation_spearman(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 1.0),
        (lazy_fixture("atts_same_linear"), {}, 1.0),
        (lazy_fixture("atts_diff"), {}, 0.0),
        (lazy_fixture("atts_half"), {}, 0.5),
        (lazy_fixture("atts_inverse"), {}, -1),
    ],
)
def test_correlation_pearson(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = correlation_pearson(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 1.0),
        (lazy_fixture("atts_same_linear"), {}, 1.0),
        (lazy_fixture("atts_diff"), {}, 0.0),
        (lazy_fixture("atts_half"), {}, 0.5),
        (lazy_fixture("atts_inverse"), {}, -1),
    ],
)
def test_correlation_kendall_tau(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = correlation_kendall_tau(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 2.0),
        (lazy_fixture("atts_half"), {}, 1.73),
    ],
)
def test_distance_euclidean(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = distance_euclidean(a=data["a"], b=data["b"])
    print(out)
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 4.0),
        (lazy_fixture("atts_half"), {}, 3.0),
    ],
)
def test_distance_manhattan(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = distance_manhattan(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 4.0),
        (lazy_fixture("atts_half"), {}, 3.0),
    ],
)
def test_distance_chebyshev(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = distance_chebyshev(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("atts_same"), {}, 0.0), (lazy_fixture("atts_diff"), {}, 1.0)],
)
def test_distance_chebyshev(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = distance_chebyshev(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_lip_same"),
            {
                "norm_numerator": distance_manhattan,
                "norm_denominator": distance_manhattan,
            },
            1.0,
        ),
        (
            lazy_fixture("atts_lip_same"),
            {
                "norm_numerator": distance_manhattan,
                "norm_denominator": distance_euclidean,
            },
            1.73,
        ),
    ],
)
def test_lipschitz_constant(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = lipschitz_constant(
        a=data["a"], b=data["b"], c=data["c"], d=data["d"], **params
    )
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 1.0),
        (lazy_fixture("atts_half"), {}, 1.0),
    ],
)
def test_abs_difference(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = abs_difference(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 0.42),
        (lazy_fixture("atts_half"), {}, 0.42),
    ],
)
def test_cosine(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    out = cosine(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_ssim_same"), {}, 1.0),
        (lazy_fixture("atts_ssim_diff"), {}, 0.0),
    ],
)
def test_ssim(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    """Calculate Structural Similarity Index Measure of two images (or explanations)."""
    out = ssim(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, 0.0),
        (lazy_fixture("atts_diff"), {}, 1.0),
        (lazy_fixture("atts_half"), {}, 1.0),
    ],
)
def test_mse(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    out = mse(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_same"), {}, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        (lazy_fixture("atts_diff"), {}, np.array([-1, -1, -1, 1])),
        (lazy_fixture("atts_half"), {}, np.array([-1, 1, -1])),
    ],
)
def test_difference(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    out = difference(a=data["a"], b=data["b"])
    assert all(out == expected), "Test failed."
