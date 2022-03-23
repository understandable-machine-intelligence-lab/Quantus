from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def atts_norm_ones():
    return np.ones((10))


@pytest.fixture
def atts_norm_fill():
    return np.array([1, 2, 3, 4, 10])


@pytest.mark.norm_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_norm_ones"), {}, 3.1622776601683795),
        (lazy_fixture("atts_norm_fill"), {}, 11.40175425099138),
    ],
)
def test_fro_norm(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    out = fro_norm(a=data)
    assert out == expected, "Test failed."


@pytest.mark.norm_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_norm_ones"), {}, 1.0),
        (lazy_fixture("atts_norm_fill"), {}, 10),
    ],
)
def test_linf_norm(data: np.ndarray, params: dict, expected: Union[float, dict, bool]):
    out = linf_norm(a=data)
    assert out == expected, "Test failed."


@pytest.mark.norm_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("atts_norm_ones"), {}, 3.1622776601683795),
        (lazy_fixture("atts_norm_fill"), {}, 11.40175425099138),
    ],
)
def test_l2_norm(data: dict, params: dict, expected: Union[float, dict, bool]):
    out = l2_norm(a=data)
    assert out == expected, "Test failed."
