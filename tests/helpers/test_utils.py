import pytest
import pickle
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def get_model(autouse=True):
    model = (
        LeNet()
    )  # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def segmentation_setup():
    return np.random.uniform(0, 0.1, size=(3, 224, 224))


@pytest.fixture
def baseline_black():
    return {"choice": "black", "img": np.random.uniform(0, 1, size=(1, 3, 224, 224))}


@pytest.fixture
def baseline_uniform():
    return {"choice": "uniform", "img": np.random.uniform(0, 1, size=(1, 3, 224, 224))}


@pytest.fixture
def baseline_random():
    return {"choice": "random", "img": np.random.uniform(0, 1, size=(1, 3, 224, 224))}


@pytest.fixture
def baseline_white():
    return {"choice": "white", "img": np.random.uniform(0, 1, size=(1, 3, 224, 224))}


@pytest.fixture
def baseline_mean():
    return {"choice": "mean", "img": np.random.uniform(0, 1, size=(1, 3, 2222, 2222))}


@pytest.mark.utils
@pytest.mark.parametrize("data,expected", [(lazy_fixture("get_model"), list)])
def test_get_layers(data: np.ndarray, expected: Union[float, dict, bool]):
    model = data
    out = get_layers(model=model)
    assert isinstance(out, expected), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("segmentation_setup"), {"segmentation_method": "slic"}, np.ndarray)],
)
def test_get_superpixel_segments(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = get_superpixel_segments(
        img=data, segmentation_method=params["segmentation_method"]
    )
    assert out is not None, "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("baseline_black"), 0.0),
        (lazy_fixture("baseline_white"), 1.0),
        (lazy_fixture("baseline_mean"), 0.5),
    ],
)
def test_get_baseline_value(data: np.ndarray, expected: Union[float, dict, bool]):
    out = get_baseline_value(choice=data["choice"], img=data["img"])
    assert round(out, 2) == expected, "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("baseline_black"), dict),
        (lazy_fixture("baseline_white"), dict),
        (lazy_fixture("baseline_mean"), dict),
    ],
)
def test_get_baseline_dict(data: np.ndarray, expected: Union[float, dict, bool]):
    out = get_baseline_dict(img=data["img"])
    assert isinstance(out, dict), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        ("PixelFlipping", "Pixel Flipping"),
        ("AUC", "AUC"),
        ("MaxSensitivity", "Max Sensitivity"),
    ],
)
def test_get_name(data: np.ndarray, expected: Union[float, dict, bool]):
    out = get_name(str=data)
    assert out == expected, "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        ({"max_steps_per_input": 4, "img_size": 28}, 196),
        ({"max_steps_per_input": 128, "img_size": 224}, 392),
        ({"max_steps_per_input": 4, "img_size": 4}, 4),
    ],
)
def test_set_features_in_step(data: np.ndarray, expected: Union[float, dict, bool]):
    out = set_features_in_step(
        max_steps_per_input=data["max_steps_per_input"], img_size=data["img_size"]
    )
    assert out == expected, "Test failed."
