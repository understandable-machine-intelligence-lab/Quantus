import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.metrics import *

@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt"), {}, True),
                                                  (lazy_fixture("none_in_gt"), {}, False),
                                                  (lazy_fixture("half_in_gt"), {}, True)])
def test_pointing_game(data: dict, params: dict, expected: Union[float, dict]):
    scores = PointingGame(**params)(model=None,
                            x_batch=data["x_batch"],
                            y_batch=data["y_batch"],
                            a_batch=data["a_batch"],
                            s_batch=data["s_batch"])

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."

@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt"), {"k": 10000}, 1.0),
                                                  (lazy_fixture("all_in_gt"), {"k": 40000}, 0.25),
                                                  (lazy_fixture("none_in_gt"), {"k": 10000}, 0.0),
                                                  (lazy_fixture("none_in_gt_zeros"), {"k": 40000}, {"min": 0.1,
                                                                                                    "max": 0.25}),
                                                  (lazy_fixture("half_in_gt_zeros"), {"k": 2500}, 0.5),
                                                  (lazy_fixture("half_in_gt_zeros"), {"k": 1250}, {"min": 0.5,
                                                                                                    "max": 1.0})])
def test_top_k_intersection(data: dict, params: dict, expected: Union[float, dict]):
    scores = TopKIntersection(**params)(model=None,
                                     x_batch=data["x_batch"],
                                     y_batch=data["y_batch"],
                                     a_batch=data["a_batch"],
                                     s_batch=data["s_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."

@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt"), {}, 1.0),
                                                  (lazy_fixture("all_in_gt_seg_bigger"), {}, {"min": 0.5, "max": 1.0}),
                                                  (lazy_fixture("none_in_gt"), {}, 0.0),
                                                  (lazy_fixture("half_in_gt"), {}, 0.5)])
def test_relevance_rank_accuracy(data: dict, params: dict, expected: Union[float, dict]):
    scores = RelevanceRankAccuracy(**params)(model=None,
                                     x_batch=data["x_batch"],
                                     y_batch=data["y_batch"],
                                     a_batch=data["a_batch"],
                                     s_batch=data["s_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."

@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt_zeros"), {}, 1.0),
                                                  (lazy_fixture("all_in_gt_seg_bigger"), {}, {"min": 0.5, "max": 1.0}),
                                                  (lazy_fixture("none_in_gt_zeros"), {}, 0.0),
                                                  (lazy_fixture("half_in_gt_zeros"), {}, 0.5)])
def test_relevance_mass_accuracy(data: dict, params: dict, expected: Union[float, dict]):
    scores = RelevanceMassAccuracy(**params)(model=None,
                                     x_batch=data["x_batch"],
                                     y_batch=data["y_batch"],
                                     a_batch=data["a_batch"],
                                     s_batch=data["s_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."

@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt"), {}, 1.0),
                                                  (lazy_fixture("all_in_gt_non_normalised"), {"normalise": False}, 1.0),
                                                  (lazy_fixture("none_in_gt_fourth"), {}, 0.33333333333333337)])
def test_auc(data: dict, params: dict, expected: Union[float, dict]):
    scores = AUC(**params)(model=None,
                                     x_batch=data["x_batch"],
                                     y_batch=data["y_batch"],
                                     a_batch=data["a_batch"],
                                     s_batch=data["s_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("all_in_gt_zeros"), {"weighted": False}, 1.0),
                                                  (lazy_fixture("all_in_gt"), {"weighted": False}, {"min": 0.8, "max": 0.85}),
                                                  (lazy_fixture("none_in_gt_zeros"), {"weighted": False}, 0.0),
                                                  (lazy_fixture("none_in_gt_zeros"), {"weighted": True}, 0.0)])
def test_attribution_localisation(data: dict, params: dict, expected: Union[float, dict]):
    scores = AttributionLocalisation(**params)(model=None,
                                     x_batch=data["x_batch"],
                                     y_batch=data["y_batch"],
                                     a_batch=data["a_batch"],
                                     s_batch=data["s_batch"])
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert all(((s > expected["min"]) & (s < expected["max"])) for s in scores), "Test failed."