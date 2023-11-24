import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.n_bins_func import *
import numpy as np


@pytest.fixture
def batch_data():
    return np.random.rand(10, 32, 32, 3)


@pytest.mark.n_bins_func
@pytest.mark.parametrize(
    "func,data",
    [
        (freedman_diaconis_rule, lazy_fixture("batch_data")),
        (scotts_rule, lazy_fixture("batch_data")),
        (square_root_choice, lazy_fixture("batch_data")),
        (sturges_formula, lazy_fixture("batch_data")),
        (rice_rule, lazy_fixture("batch_data")),
    ],
)
def test_n_bins_func(func, data: np.ndarray):
    n_bins = func(data)
    print(n_bins)
    assert isinstance(n_bins, int), "Output should be an integer."
    assert n_bins > 0, "Number of bins should be positive."
