import numpy as np
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
    return np.random.uniform(0, 0.1, size=(224, 224, 3))


@pytest.fixture
def baseline_none():
    return {"choice": None, "img": np.random.uniform(0, 1, size=(1, 3, 224, 224))}


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


@pytest.fixture
def mock_input_torch_array():
    return {"x": np.zeros((1, 1, 28, 28))}


@pytest.fixture
def mock_input_tf_array():
    return {"x": np.zeros((1, 28, 28, 1))}


@pytest.fixture
def mock_same_input():
    return {"x": np.zeros((1, 28, 28, 28))}


@pytest.fixture
def mock_mismatch_input():
    return {"x": np.zeros((1, 1, 2, 3))}


def random_input(c_in, imsize):
    return np.random.uniform(0, 1, size=(c_in, imsize, imsize))


def random_kernel(c_in, c_out, groups, ksize):
    return np.random.uniform(0, 1, size=(c_out, c_in // groups, ksize, ksize))


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("segmentation_setup"),
            {"segmentation_method": "slic"},
            {"type": np.ndarray},
        ),
        (
            lazy_fixture("segmentation_setup"),
            {"segmentation_method": "felzenszwalb"},
            {"type": np.ndarray},
        ),
        (
            lazy_fixture("segmentation_setup"),
            {"segmentation_method": "ERROR"},
            {"exception": AssertionError},
        ),
    ],
)
def test_get_superpixel_segments(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_superpixel_segments(
                img=data, segmentation_method=params["segmentation_method"]
            )
        return
    out = get_superpixel_segments(
        img=data, segmentation_method=params["segmentation_method"]
    )
    assert isinstance(out, expected["type"]), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("baseline_black"), {"value": 0.0}),
        (lazy_fixture("baseline_white"), {"value": 1.0}),
        (lazy_fixture("baseline_mean"), {"value": 0.5}),
        (lazy_fixture("baseline_none"), {"exception": AssertionError}),
    ],
)
def test_get_baseline_value(data: np.ndarray, expected: Union[float, dict, bool]):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_baseline_value(choice=data["choice"], img=data["img"])
        return
    out = get_baseline_value(choice=data["choice"], img=data["img"])
    assert round(out, 2) == expected["value"], "Test failed."


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
        ({"perturb_patch_sizes": [1, 2, 3, 4], "img_size": 1}, [1]),
        ({"perturb_patch_sizes": [1, 2, 3, 4], "img_size": 8}, [1, 2, 4]),
    ],
)
def test_filter_compatible_patch_sizes(
    data: np.ndarray, expected: Union[float, dict, bool]
):
    out = filter_compatible_patch_sizes(
        perturb_patch_sizes=data["perturb_patch_sizes"], img_size=data["img_size"]
    )
    assert out == expected, "Test failed."


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


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("mock_input_tf_array"), {"value": False}),
        (lazy_fixture("mock_input_torch_array"), {"value": True}),
        (lazy_fixture("mock_same_input"), {"exception": ValueError}),
        (lazy_fixture("mock_mismatch_input"), {"exception": ValueError}),
    ],
)
def test_get_channel_first(data: np.ndarray, expected: Union[float, dict, bool]):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_channel_first(data["x"])
        return
    out = get_channel_first(data["x"])
    assert out == expected["value"], "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("mock_input_tf_array"),
            {"channel_first": False},
            np.zeros((1, 1, 28, 28)),
        ),
        (
            lazy_fixture("mock_input_torch_array"),
            {"channel_first": True},
            np.zeros((1, 1, 28, 28)),
        ),
    ],
)
def test_get_channel_first_batch(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = get_channel_first_batch(data["x"], params["channel_first"])
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("mock_input_tf_array"),
            {"channel_first": False},
            np.zeros((1, 28, 28, 1)),
        ),
        (
            lazy_fixture("mock_input_torch_array"),
            {"channel_first": True},
            np.zeros((1, 28, 28, 1)),
        ),
    ],
)
def test_get_channel_last_batch(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = get_channel_last_batch(data["x"], params["channel_first"])
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "model,params,expected",
    [
        (
            lazy_fixture("load_mnist_model_tf"),
            {"channel_first": False},
            {"type": TensorFlowModel},
        ),
        (
            lazy_fixture("load_mnist_model"),
            {"channel_first": True},
            {"type": PyTorchModel},
        ),
        (
            None,
            {"channel_first": True},
            {"exception": ValueError},
        ),
    ],
)
def test_get_wrapped_model(
    model: ModelInterface,
    params: dict,
    expected: Union[float, dict, bool],
):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = get_wrapped_model(model, params["channel_first"])
        return
    out = get_wrapped_model(model, params["channel_first"])
    if "type" in expected:
        isinstance(out, expected["type"]), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "c_in": 3,
                "c_out": 3,
                "imsize": 224,
                "kgroups": 3,
                "ksize": 8,
                "stride": 1,
                "padding": 0,
                "groups": 3,
            },
            {"shape": (3, 217, 217)},
        ),
        (
            {
                "c_in": 3,
                "c_out": 3,
                "imsize": 224,
                "kgroups": 3,
                "ksize": 8,
                "stride": 1,
                "padding": 3,
                "groups": 3,
            },
            {"shape": (3, 223, 223)},
        ),
        (
            {
                "c_in": 3,
                "c_out": 3,
                "imsize": 224,
                "kgroups": 3,
                "ksize": 8,
                "stride": 3,
                "padding": 0,
                "groups": 3,
            },
            {"shape": (3, 73, 73)},
        ),
        (
            {
                "c_in": 6,
                "c_out": 3,
                "imsize": 224,
                "kgroups": 3,
                "ksize": 8,
                "stride": 1,
                "padding": 0,
                "groups": 3,
            },
            {"shape": (3, 217, 217)},
        ),
        (
            {
                "c_in": 6,
                "c_out": 3,
                "imsize": 224,
                "kgroups": 2,
                "ksize": 8,
                "stride": 1,
                "padding": 0,
                "groups": 3,
            },
            {"exception": AssertionError},
        ),
    ],
)
def test_conv2D_numpy(
    params: dict,
    expected: Union[float, dict, bool],
):
    input = random_input(params["c_in"], params["imsize"])
    kernel = random_kernel(
        params["c_in"], params["c_out"], params["kgroups"], params["ksize"]
    )
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = conv2D_numpy(
                input, kernel, params["stride"], params["padding"], params["groups"]
            )
        return
    out = conv2D_numpy(
        input, kernel, params["stride"], params["padding"], params["groups"]
    )
    if "shape" in expected:
        assert expected["shape"] == out.shape, "Test failed."
