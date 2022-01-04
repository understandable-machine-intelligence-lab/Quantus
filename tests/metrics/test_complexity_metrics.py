import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


@pytest.mark.complexity
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("almost_uniform"),
            {"normalise": True, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("almost_uniform"),
            {"normalise": False, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_sparseness(data: dict, params: dict, expected: Union[float, dict, bool]):
    scores = Sparseness(**params)(
        model=None,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(
            ((s > expected["min"]) & (s < expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.complexity
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("almost_uniform"),
            {"normalise": True, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("almost_uniform"),
            {"normalise": False, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_complexity(data: dict, params: dict, expected: Union[float, dict, bool]):
    scores = Complexity(**params)(
        model=None,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
    )
    assert scores is not None, "Test failed."


@pytest.mark.complexity
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("almost_uniform"),
            {"normalise": True, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
        (
            lazy_fixture("almost_uniform"),
            {"normalise": False, "disable_warnings": True},
            {"max": 1.0, "min": 0.0},
        ),
    ],
)
def test_effective_complexity(
    data: dict, params: dict, expected: Union[float, dict, bool]
):
    scores = EffectiveComplexity(**params)(
        model=None,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
    )
    assert scores is not None, "Test failed."
