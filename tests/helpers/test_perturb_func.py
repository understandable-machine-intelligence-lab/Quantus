import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert"), {}, True)])
def test_gaussian_blur(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = gaussian_blur(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_1d"), {}, True)])
def test_gaussian_noise(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = gaussian_noise(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_1d"), {"index": [0, 2], "fixed_values": 1.0}, True)])
def test_baseline_replacement_by_indices(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = baseline_replacement_by_indices(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_3d"), {"patch_size": 4,
                                                                                   "nr_channels": 3,
                                                                                   "perturb_baseline": "black",
                                                                                   "top_left_y": 0,
                                                                                   "top_left_x": 0}, True)])
def test_baseline_replacement_by_patch(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = baseline_replacement_by_patch(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_1d"), {"perturb_radius": 0.02}, True)])
def test_uniform_sampling(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = uniform_sampling(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_3d"), {"perturb_angle": 30}, True)])
def test_rotation(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = rotation(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_3d"), {"perturb_dx": 20}, True)])
def test_translation_x_direction(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = translation_x_direction(img=data, **params)
    assert any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize("data,params,expected", [(lazy_fixture("input_pert_3d"), {"perturb_dx": 20}, True)])
def test_translation_y_direction(data: dict,
                       params: dict,
                       expected: Union[float, dict]):
    out = translation_y_direction(img=data, **params)
    assert any(out != data) == expected, "Test failed."


