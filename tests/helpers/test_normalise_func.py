import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.normalise_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_normalise"), {},
                                                   np.array([0.2,  0.4,  0.6,  0.8,  0.8,  1.0, -0.2]))])
def test_normalise_by_max(data: dict,
                          params: dict,
                          expected: Union[float, dict]):
    out = normalise_by_max(a=data)
    assert all(out == expected), "Test failed."
