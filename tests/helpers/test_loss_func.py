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


@pytest.mark.loss_func
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
