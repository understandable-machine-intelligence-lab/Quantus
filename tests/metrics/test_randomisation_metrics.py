import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


# TODO. Finish test.
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
        (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0),
    ],
)
def test_model_parameter_randomsiation(
    data: dict, params: dict, expected: Union[float, dict]
):
    scores = ModelParameterRandomisation(**params)(
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
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
        (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0),
    ],
)
def test_non_random_logit(data: dict, params: dict, expected: Union[float, dict]):
    scores = RandomLogit(**params)(
        model=model,
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
