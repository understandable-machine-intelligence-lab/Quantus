import pytest
import numpy as np
from quantus.functions.complexity_func import entropy, gini_coeffiient, discrete_entropy


@pytest.fixture
def array_data():
    return np.random.rand(1, 32, 32), np.random.rand(1, 32, 32)


@pytest.mark.complexity_func
def test_entropy(array_data):
    a, x = array_data
    result = entropy(a, x)
    assert isinstance(result, float), "Output should be a float."


@pytest.mark.complexity_func
def test_gini_coefficient(array_data):
    a, x = array_data
    result = gini_coeffiient(a, x)
    assert isinstance(result, float), "Output should be a float."
    assert 0 <= result <= 1, "Gini coefficient should be in the range [0, 1]."


@pytest.mark.complexity_func
@pytest.mark.parametrize("n_bins", [10, 50, 100])
def test_discrete_entropy(array_data, n_bins):
    a, x = array_data
    result = discrete_entropy(a, x, n_bins=n_bins)
    assert isinstance(result, float), "Output should be a float."
