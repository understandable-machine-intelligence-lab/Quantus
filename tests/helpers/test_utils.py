import pickle
from typing import Any, Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers.models import LeNet
from ...quantus.helpers.utils import *

@pytest.fixture
def segmentation_setup():
    return np.random.uniform(0, 0.1, size=(224, 224, 3))


@pytest.fixture
def baseline_none_1d():
    return {"value": None, "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_none_3d():
    return {"value": None, "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


@pytest.fixture
def baseline_black_1d():
    return {"value": "black", "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_black_3d():
    return {"value": "black", "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


@pytest.fixture
def baseline_uniform_1d():
    return {"value": "uniform", "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_uniform_3d():
    return {"value": "uniform", "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


@pytest.fixture
def baseline_random_1d():
    return {"value": "random", "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_random_3d():
    return {"value": "random", "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


@pytest.fixture
def baseline_white_1d():
    return {"value": "white", "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_white_3d():
    return {"value": "white", "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


@pytest.fixture
def baseline_mean_1d():
    return {"value": "mean", "arr": np.random.uniform(0, 1, size=(128,))}


@pytest.fixture
def baseline_mean_3d():
    return {"value": "mean", "arr": np.random.uniform(0, 1, size=(128, 64, 32))}


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
            {"exception": ValueError},
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
    "data,shape,expected",
    [
        (lazy_fixture("baseline_black_1d"), {"return_shape": (1,)}, {"value": 0.0}),
        (lazy_fixture("baseline_black_1d"), {"return_shape": (3, 4)}, {"value": 0.0}),
        (lazy_fixture("baseline_black_3d"), {"return_shape": (1,)}, {"value": 0.0}),
        (lazy_fixture("baseline_black_3d"), {"return_shape": (3, 4)}, {"value": 0.0}),
        (lazy_fixture("baseline_white_1d"), {"return_shape": (1,)}, {"value": 1.0}),
        (lazy_fixture("baseline_white_1d"), {"return_shape": (3, 4)}, {"value": 1.0}),
        (lazy_fixture("baseline_white_3d"), {"return_shape": (1,)}, {"value": 1.0}),
        (lazy_fixture("baseline_white_3d"), {"return_shape": (3, 4)}, {"value": 1.0}),
        (lazy_fixture("baseline_mean_1d"), {"return_shape": (1,)}, {"value": 0.5}),
        (lazy_fixture("baseline_mean_1d"), {"return_shape": (3, 4)}, {"value": 0.5}),
        (lazy_fixture("baseline_mean_3d"), {"return_shape": (1,)}, {"value": 0.5}),
        (lazy_fixture("baseline_mean_3d"), {"return_shape": (3, 4)}, {"value": 0.5}),
        (lazy_fixture("baseline_none_1d"), {"return_shape": (1,)}, {"exception": ValueError}),
        (lazy_fixture("baseline_none_1d"), {"return_shape": (3, 4)}, {"exception": ValueError}),
        (lazy_fixture("baseline_none_3d"), {"return_shape": (1,)}, {"exception": ValueError}),
        (lazy_fixture("baseline_none_3d"), {"return_shape": (3, 4)}, {"exception": ValueError}),
    ],
)
def test_get_baseline_value(data: np.ndarray, shape: Tuple, expected: Union[float, dict, bool]):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_baseline_value(value=data["value"], return_shape=shape["return_shape"], arr=data["arr"])
        return
    out = get_baseline_value(value=data["value"], return_shape=shape["return_shape"], arr=data["arr"])
    assert out == pytest.approx(expected["value"], rel=1e-1, abs=1e-1) and out.shape == shape["return_shape"], "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,expected",
    [
        (lazy_fixture("baseline_black_3d"), dict),
        (lazy_fixture("baseline_black_3d"), dict),
        (lazy_fixture("baseline_white_3d"), dict),
        (lazy_fixture("baseline_white_3d"), dict),
        (lazy_fixture("baseline_mean_3d"), dict),
        (lazy_fixture("baseline_mean_3d"), dict),
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
        (
            {
                "c_in": 1,
                "c_out": 1,
                "imsize": 28,
                "kgroups": 1,
                "ksize": 15,
                "stride": 1,
                "padding": 0,
                "groups": 1,
                "pad_output": True,
            },
            {"shape": (1, 28, 28)},
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
                "coords": 0,
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), )},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, ),
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), )},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), slice(0, 4, None))},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), slice(0, 4, None))},
        ),
        (
            {
                "patch_size": (4, 6),
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), slice(0, 6, None))},
        ),
        (
            {
                "patch_size": 10,
                "coords": (1, 2),
                "expand_first_dim": False,
            },
            {"value": (slice(1, 11, None), slice(2, 12, None))},
        ),
        (
            {
                "patch_size": (10, 5),
                "coords": (1, 2),
                "expand_first_dim": False,
            },
            {"value": (slice(1, 11, None), slice(2, 7, None))},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0),
                "expand_first_dim": True,
            },
            {"value": (slice(None, None, None), slice(0, 4, None), slice(0, 4, None))},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0, 0),
                "expand_first_dim": False,
            },
            {"value": (slice(0, 4, None), slice(0, 4, None), slice(0, 4, None))},
        ),
        (
            {
                "patch_size": (4, 4, 4),
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0, 0, 0),
                "expand_first_dim": False,
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0, ),
                "expand_first_dim": False,
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": np.ones((4, 4)),
                "coords": (0, 0),
                "expand_first_dim": False,
            },
            {"exception": ValueError},
        ),
    ],
)
def test_create_patch_slice(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = create_patch_slice(**params)
        return

    out = create_patch_slice(**params)
    
    assert all(out_slice == expected_slice
               for out_slice, expected_slice in zip(out, expected["value"])
    ), f"Slices not equal. {out_slice} != {expected_slice}"
    assert all(isinstance(out_slice.start, int) or out_slice.start is None
               for out_slice in out
    ), f"Not all slice starts are integers/None. {out}"
    assert all(isinstance(out_slice.stop, int) or out_slice.stop is None
               for out_slice in out
    ), f"Not all slice stops are integers/None. {out}"
    assert all(isinstance(out_slice.step, int) or out_slice.step is None
               for out_slice in out
    ), f"Not all slice steps are integers/None. {out}"


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "a_batch": np.ones((64, 128)),
                "x_batch": np.ones((64, 3, 128)),
            },
            {"value": np.ones((64, 1, 128))},
        ),
        (
            {
                "a_batch": np.ones((64, 128, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 1, 128, 128))},
        ),
        (
            {
                "a_batch": np.ones((64, 1, 128, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 1, 128, 128))},
        ),
        (
            {
                "a_batch": np.ones((64, 3, 128, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 3, 128, 128))},
        ),
        (
            {
                "a_batch": np.ones((64, 3, 128, 128)),
                "x_batch": np.ones((32, 3, 128, 128)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "a_batch": np.ones((64, 3, 128, 128)),
                "x_batch": np.ones((64, 3, 128)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "a_batch": np.ones((64, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"exception": ValueError},
        ),
    ],
)
def test_expand_attribution_channel(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = expand_attribution_channel(**params)
        return

    out = expand_attribution_channel(**params)
    assert out.shape == expected["value"].shape
    assert (out == expected["value"]).any()


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "patch_size": 4,
                "shape": (16, ),
            },
            {"value": 4},
        ),
        (
            {
                "patch_size": (4, ),
                "shape": (16, ),
            },
            {"value": 4},
        ),
        (
            {
                "patch_size": (4, 4),
                "shape": (16, ),
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": 4,
                "shape": (16, 16),
            },
            {"value": 16},
        ),
        (
            {
                "patch_size": (4, 4),
                "shape": (16, 16),
            },
            {"value": 16},
        ),
        (
            {
                "patch_size": (4, 2),
                "shape": (16, 16),
            },
            {"value": 32},
        ),
        (
            {
                "patch_size": (4, 4, 4),
                "shape": (16, 16),
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": (4, ),
                "shape": (16, 16, 16),
            },
            {"value": 64},
        ),
        (
            {
                "patch_size": (4, 4),
                "shape": (16, 16, 16),
            },
            {"exception": ValueError},
        ),
    ],
)
def test_get_nr_patches(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = get_nr_patches(**params)
        return

    out = get_nr_patches(**params)
    assert out == expected["value"]

@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 2, 3, 4, 5, 6)),
            },
            {"value": [0, 1, 2, 3, 4]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 3, 4, 5, 6)),
            },
            {"value": [1, 2, 3, 4]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 4, 5, 6)),
            },
            {"value": [2, 3, 4]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 2, 3, 4)),
            },
            {"value": [0, 1, 2]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30,)),
            },
            {"value": []},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 9, 9, 9, 9, 9, 9)),
            },
            {"exception": ValueError},
        ),
    ],
)
def test_infer_attribution_axes(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = infer_attribution_axes(**params)
        return

    out = infer_attribution_axes(**params)
    assert all([a == b for a, b in list(zip(out, expected["value"]))])


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": 5,
                "indexed_axes": [0, 1, 2, 3, 4],
            },
            {"value": (0, 0, 0, 0, 5)},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": [5, 19],
                "indexed_axes": [2, 3, 4],
            },
            {"value": (slice(None), slice(None), np.array([0, 0]), np.array([0, 3]), np.array([5, 1]))},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": (np.array([1, 1]), np.array([0, 2])),
                "indexed_axes": [0, 1],
            },
            {"value": (np.array([1, 1]), np.array([0, 2]), slice(None), slice(None), slice(None))},
        ),
        (
            {
                "arr": np.ones((2,)),
                "indices": [1],
                "indexed_axes": [0, 1, 2, 3],
            },
            {"exception": AssertionError},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": (np.array([1, 1]), np.array([0, 2])),
                "indexed_axes": [0],
            },
            {"exception": ValueError},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": [1],
                "indexed_axes": [0, 2],
            },
            {"exception": AssertionError},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": [1],
                "indexed_axes": [2, 3],
            },
            {"exception": AssertionError},
        ),
    ],
)
def test_expand_indices(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = expand_indices(**params)
        return

    out = expand_indices(**params)
    assert all([np.all(a == b) for a, b in list(zip(out, expected["value"]))])


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": [0, 1, 2, 3, 4],
            },
            {"value": ()},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": np.array([0, 1]),
            },
            {"value": (4, 5, 6)},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": np.array([1, 2, 3, 4]),
            },
            {"value": (2,)},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": np.array([0, 2]),
            },
            {"exception": AssertionError},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": np.array([1, 2]),
            },
            {"exception": AssertionError},
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "axes": np.array([0, 1, 2, 3, 4, 5]),
            },
            {"exception": AssertionError},
        ),
    ]
)
def test_get_leftover_shape(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = get_leftover_shape(**params)
        return

    out = get_leftover_shape(**params)
    assert all([np.all(a == b) for a, b in list(zip(out, expected["value"]))])