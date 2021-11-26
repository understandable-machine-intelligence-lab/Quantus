import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.norm_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_norm_ones"), {}, 3.1622776601683795),
                                                  (lazy_fixture("atts_norm_fill"), {}, 11.40175425099138)])
def test_fro_norm(data: dict,
                  params: dict,
                  expected: Union[float, dict]):
    out = fro_norm(a=data)
    assert out == expected, "Test failed."


@pytest.mark.norm_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_norm_ones"), {}, 1.0),
                                                  (lazy_fixture("atts_norm_fill"), {}, 10)])
def test_linf_norm(data: dict,
                   params: dict,
                   expected: Union[float, dict]):
    out = linf_norm(a=data)
    assert out == expected, "Test failed."

'''
def test_l2_norm(data: dict,
                          params: dict,
                          expected: Union[float, dict]):
    out = linf_norm(a=data)
    assert out == expected, "Test failed."


def test_l1_norm(data: dict,
                          params: dict,
                          expected: Union[float, dict]):
    out = linf_norm(a=data)
    assert out == expected, "Test failed."
'''