import numpy as np
import pytest
import pickle
from typing import Any, Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers.models import LeNet
from ...quantus.helpers.utils import *


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
def baseline_none_1d():
    return {"choice": None, "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_none_2d():
    return {"choice": None, "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_black_1d():
    return {"choice": "black", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_black_2d():
    return {"choice": "black", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_uniform_1d():
    return {"choice": "uniform", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_uniform_2d():
    return {"choice": "uniform", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_random_1d():
    return {"choice": "random", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_random_2d():
    return {"choice": "random", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_white_1d():
    return {"choice": "white", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_white_2d():
    return {"choice": "white", "arr": np.random.uniform(0, 1, size=(1, 3, 224))}


@pytest.fixture
def baseline_mean_1d():
    return {"choice": "mean", "arr": np.random.uniform(0, 1, size=(1, 3, 2222))}


@pytest.fixture
def baseline_mean_2d():
    return {"choice": "mean", "arr": np.random.uniform(0, 1, size=(1, 3, 2222, 2222))}


@pytest.fixture
def mock_input_torch_array_1d():
    return {"x": np.zeros((1, 1, 28))}


@pytest.fixture
def mock_input_tf_array_1d():
    return {"x": np.zeros((1, 28, 1))}


@pytest.fixture
def mock_input_torch_array_2d():
    return {"x": np.zeros((1, 1, 28, 32))}


@pytest.fixture
def mock_input_tf_array_2d():
    return {"x": np.zeros((1, 28, 32, 1))}


@pytest.fixture
def mock_input_torch_array_2d_squared():
    return {"x": np.zeros((1, 1, 28, 28))}


@pytest.fixture
def mock_input_tf_array_2d_squared():
    return {"x": np.zeros((1, 28, 28, 1))}


@pytest.fixture
def mock_input_torch_array_3d():
    return {"x": np.zeros((1, 1, 28, 28, 28))}


@pytest.fixture
def mock_input_tf_array_3d():
    return {"x": np.zeros((1, 28, 28, 28, 1))}


@pytest.fixture
def mock_unbatched_input_1d():
    return {"x": np.zeros((28))}


@pytest.fixture
def mock_same_input_1d():
    return {"x": np.zeros((1, 28, 28))}


@pytest.fixture
def mock_same_input_2d():
    return {"x": np.zeros((1, 28, 28, 28))}


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
        (lazy_fixture("baseline_black_1d"), {"value": 0.0}),
        (lazy_fixture("baseline_black_2d"), {"value": 0.0}),
        (lazy_fixture("baseline_white_1d"), {"value": 1.0}),
        (lazy_fixture("baseline_white_2d"), {"value": 1.0}),
        (lazy_fixture("baseline_mean_1d"), {"value": 0.5}),
        (lazy_fixture("baseline_mean_2d"), {"value": 0.5}),
        (lazy_fixture("baseline_none_1d"), {"exception": AssertionError}),
        (lazy_fixture("baseline_none_2d"), {"exception": AssertionError}),
    ],
)
def test_get_baseline_value(data: np.ndarray, expected: Union[float, dict, bool]):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_baseline_value(choice=data["choice"], arr=data["arr"])
        return
    out = get_baseline_value(choice=data["choice"], arr=data["arr"])
    assert out == pytest.approx(expected["value"], rel=1e-1, abs=1e-1), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("baseline_black_2d"), dict),
        (lazy_fixture("baseline_black_2d"), dict),
        (lazy_fixture("baseline_white_2d"), dict),
        (lazy_fixture("baseline_white_2d"), dict),
        (lazy_fixture("baseline_mean_2d"), dict),
        (lazy_fixture("baseline_mean_2d"), dict),
    ],
)
def test_get_baseline_dict(data: np.ndarray, expected: Union[float, dict, bool]):
    out = get_baseline_dict(arr=data["arr"])
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
        ({"max_steps_per_input": 4, "input_shape": (28, 28)}, 196),
        ({"max_steps_per_input": 128, "input_shape": (224, 224)}, 392),
        ({"max_steps_per_input": 4, "input_shape": (4, 4)}, 4),
        ({"max_steps_per_input": 4, "input_shape": (28,)}, 7),
        ({"max_steps_per_input": 128, "input_shape": (256,)}, 2),
        ({"max_steps_per_input": 4, "input_shape": (4,)}, 1),
    ],
)
def test_get_features_in_step(data: np.ndarray, expected: Union[float, dict, bool]):
    out = get_features_in_step(
        max_steps_per_input=data["max_steps_per_input"],
        input_shape=data["input_shape"],
    )
    assert out == expected, "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("mock_input_tf_array_1d"), {"value": False}),
        (lazy_fixture("mock_input_torch_array_1d"), {"value": True}),
        (lazy_fixture("mock_input_tf_array_2d"), {"value": False}),
        (lazy_fixture("mock_input_torch_array_2d"), {"value": True}),
        (lazy_fixture("mock_input_tf_array_2d_squared"), {"value": False}),
        (lazy_fixture("mock_input_torch_array_2d_squared"), {"value": True}),
        (lazy_fixture("mock_input_tf_array_3d"), {"exception": ValueError}),
        (lazy_fixture("mock_input_torch_array_3d"), {"exception": ValueError}),
        (lazy_fixture("mock_unbatched_input_1d"), {"exception": ValueError}),
        (lazy_fixture("mock_same_input_1d"), {"exception": ValueError}),
        (lazy_fixture("mock_same_input_2d"), {"exception": ValueError}),
    ],
)
def test_infer_channel_first(data: np.ndarray, expected: Union[float, dict, bool]):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            infer_channel_first(data["x"])
        return
    out = infer_channel_first(data["x"])
    assert out == expected["value"], "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("mock_input_tf_array_1d"),
            {"channel_first": False},
            np.zeros((1, 1, 28)),
        ),
        (
            lazy_fixture("mock_input_torch_array_1d"),
            {"channel_first": True},
            np.zeros((1, 1, 28)),
        ),
        (
            lazy_fixture("mock_input_tf_array_2d"),
            {"channel_first": False},
            np.zeros((1, 1, 28, 32)),
        ),
        (
            lazy_fixture("mock_input_torch_array_2d"),
            {"channel_first": True},
            np.zeros((1, 1, 28, 32)),
        ),
    ],
)
def test_make_channel_first(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = make_channel_first(data["x"], params["channel_first"])
    assert np.array_equal(out, expected), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("mock_input_tf_array_1d"),
            {"channel_first": False},
            np.zeros((1, 28, 1)),
        ),
        (
            lazy_fixture("mock_input_torch_array_1d"),
            {"channel_first": True},
            np.zeros((1, 28, 1)),
        ),
        (
            lazy_fixture("mock_input_tf_array_2d"),
            {"channel_first": False},
            np.zeros((1, 28, 32, 1)),
        ),
        (
            lazy_fixture("mock_input_torch_array_2d"),
            {"channel_first": True},
            np.zeros((1, 28, 32, 1)),
        ),
    ],
)
def test_make_channel_last(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = make_channel_last(data["x"], params["channel_first"])
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
                "pad_output": False,
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
                "pad_output": False,
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
                "pad_output": False,
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
                "pad_output": False,
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
                "pad_output": False,
            },
            {"exception": AssertionError},
        ),
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
                "pad_output": True,
            },
            {"shape": (3, 224, 224)},
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
                "pad_output": True,
            },
            {"exception": NotImplementedError},
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
                "pad_output": True,
            },
            {"exception": NotImplementedError},
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
                "pad_output": True,
            },
            {"shape": (3, 224, 224)},
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
                x=input,
                kernel=kernel,
                stride=params["stride"],
                padding=params["padding"],
                groups=params["groups"],
                pad_output=params["pad_output"],
            )
        return
    out = conv2D_numpy(
        x=input,
        kernel=kernel,
        stride=params["stride"],
        padding=params["padding"],
        groups=params["groups"],
        pad_output=params["pad_output"],
    )
    if "shape" in expected:
        assert expected["shape"] == out.shape, "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "patch_size": 4,
                "coords": (0, ),
                "expand_first_dim": False,
            },
            (slice(0, 4, None), ),
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            (slice(0, 4, None), slice(0, 4, None)),
        ),
        (
            {
                "patch_size": 10,
                "coords": (1, 2),
                "expand_first_dim": False,
            },
            (slice(1, 11, None), slice(2, 12, None)),
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0),
                "expand_first_dim": True,
            },
            (slice(None, None, None), slice(0, 4, None), slice(0, 4, None)),
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0, 0),
                "expand_first_dim": False,
            },
            (slice(0, 4, None), slice(0, 4, None), slice(0, 4, None)),
        ),
    ],
)
def test_create_patch_slice(params: dict, expected: Any):
    out = create_patch_slice(**params)
    assert out == expected
