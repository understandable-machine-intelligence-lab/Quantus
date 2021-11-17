import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_faithfulness_correlation(data: dict, params: dict, expected: Union[float, dict]):
    scores = FaithfulnessCorrelation(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_faithfulness_estimate(data: dict, params: dict, expected: Union[float, dict]):
    scores = FaithfulnessEstimate(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_monotonicity_arya(data: dict, params: dict, expected: Union[float, dict]):
    scores = MonotonicityArya(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_monotonicity_nguyen(data: dict, params: dict, expected: Union[float, dict]):
    scores = MonotonicityNguyen(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_pixel_flipping(data: dict, params: dict, expected: Union[float, dict]):
    scores = PixelFlipping(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_region_segmentation(data: dict, params: dict, expected: Union[float, dict]):
    scores = RegionPerturbation(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_selectivity(data: dict, params: dict, expected: Union[float, dict]):
    scores = Selectivity(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_sensitivity_n(data: dict, params: dict, expected: Union[float, dict]):
    scores = SensitivityN(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_irof(data: dict, params: dict, expected: Union[float, dict]):
    scores = IROF(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.faithfulness
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("almost_uniform"), {"normalise": True}, 1.0),
                                                  (lazy_fixture("almost_uniform"), {"normalise": False}, 1.0)])
def test_irof(data: dict, params: dict, expected: Union[float, dict]):
    scores = IROF(**params)(model=None,
                                  x_batch=data["x_batch"],
                                  y_batch=data["y_batch"],
                                  a_batch=data["a_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."

