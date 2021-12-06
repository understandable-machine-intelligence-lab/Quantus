import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


# TODO. Finish test.
@pytest.mark.complexity
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
        (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0),
    ],
)
def test_sparseness(data: dict, params: dict, expected: Union[float, dict]):
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
        (lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
        (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0),
    ],
)
def test_complexity(data: dict, params: dict, expected: Union[float, dict]):
    scores = Complexity(**params)(
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


# TODO. Finish test.
@pytest.mark.complexity
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
        (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0),
    ],
)
def test_effective_complexity(data: dict, params: dict, expected: Union[float, dict]):
    scores = EffectiveComplexity(**params)(
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
