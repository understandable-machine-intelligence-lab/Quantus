import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 1.0),
                                                  (lazy_fixture("atts_same_linear"), {}, 1.0),
                                                  (lazy_fixture("atts_diff"), {}, 0.0),
                                                  (lazy_fixture("atts_half"), {}, 0.5),
                                                  (lazy_fixture("atts_inverse"), {}, -1)])
def test_correlation_spearman(data: dict,
                              params: dict,
                              expected: Union[float, dict]):
    out = correlation_spearman(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 1.0),
                                                  (lazy_fixture("atts_same_linear"), {}, 1.0),
                                                  (lazy_fixture("atts_diff"), {}, 0.0),
                                                  (lazy_fixture("atts_half"), {}, 0.5),
                                                  (lazy_fixture("atts_inverse"), {}, -1)])
def test_correlation_pearson(data: dict, params: dict, expected: Union[float, dict]):
    out = correlation_pearson(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 1.0),
                                                  (lazy_fixture("atts_same_linear"), {}, 1.0),
                                                  (lazy_fixture("atts_diff"), {}, 0.0),
                                                  (lazy_fixture("atts_half"), {}, 0.5),
                                                  (lazy_fixture("atts_inverse"), {}, -1)])
def test_correlation_kendall_tau(data: dict, params: dict, expected: Union[float, dict]):
    out = correlation_kendall_tau(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 2.0),
                                                  (lazy_fixture("atts_half"), {}, 1.73)])
def test_distance_euclidean(data: dict, params: dict, expected: Union[float, dict]):
    out = distance_euclidean(a=data["a"], b=data["b"])
    print(out)
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 4.0),
                                                  (lazy_fixture("atts_half"), {}, 3.0)])
def test_distance_manhattan(data: dict, params: dict, expected: Union[float, dict]):
    out = distance_manhattan(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 4.0),
                                                  (lazy_fixture("atts_half"), {}, 3.0)])
def test_distance_chebyshev(data: dict, params: dict, expected: Union[float, dict]):
    out = distance_chebyshev(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 1.0)])
def test_distance_chebyshev(data: dict, params: dict, expected: Union[float, dict]):
    out = distance_chebyshev(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_lip_same"),
                                                   {"norm_numerator": distance_manhattan,
                                                    "norm_denominator": distance_manhattan}, 1.0),
                                                  (lazy_fixture("atts_lip_same"),
                                                   {"norm_numerator": distance_manhattan,
                                                    "norm_denominator": distance_euclidean}, 1.73),
                                                 ])
def test_lipschitz_constant(data: dict, params: dict, expected: Union[float, dict]):
    out = lipschitz_constant(a=data["a"], b=data["b"], c=data["c"], d=data["d"], **params)
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 1.0),
                                                  (lazy_fixture("atts_half"), {}, 1.0)])
def test_abs_difference(data: dict, params: dict, expected: Union[float, dict]):
    out = abs_difference(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 0.42),
                                                  (lazy_fixture("atts_half"), {}, 0.42)])
def test_cosine(data: dict, params: dict, expected: Union[float, dict]):
    out = cosine(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_ssim_same"), {}, 1.0),
                                                  (lazy_fixture("atts_ssim_diff"), {}, 0.0)])
def test_ssim(data: dict, params: dict, expected: Union[float, dict]):
    """Calculate Structural Similarity Index Measure of two images (or explanations)."""
    out = ssim(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, 0.0),
                                                  (lazy_fixture("atts_diff"), {}, 1.0),
                                                  (lazy_fixture("atts_half"), {}, 1.0)])
def test_mse(data: dict, params: dict, expected: Union[float, dict]):
    out = mse(a=data["a"], b=data["b"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.similar_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("atts_same"), {}, np.array([0, 0, 0, 0, 0,
                                                                                            0, 0, 0, 0, 0])),
                                                  (lazy_fixture("atts_diff"), {}, np.array([-1, -1, -1, 1])),
                                                  (lazy_fixture("atts_half"), {}, np.array([-1, 1, -1]))])
def test_difference(data: dict, params: dict, expected: Union[float, dict]):
    out = difference(a=data["a"], b=data["b"])
    assert all(out == expected), "Test failed."

