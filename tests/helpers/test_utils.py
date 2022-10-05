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


def random_array(shape):
    return np.random.uniform(0, 1, size=shape)


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
        (
            lazy_fixture("baseline_none_1d"),
            {"return_shape": (1,)},
            {"exception": ValueError},
        ),
        (
            lazy_fixture("baseline_none_1d"),
            {"return_shape": (3, 4)},
            {"exception": ValueError},
        ),
        (
            lazy_fixture("baseline_none_3d"),
            {"return_shape": (1,)},
            {"exception": ValueError},
        ),
        (
            lazy_fixture("baseline_none_3d"),
            {"return_shape": (3, 4)},
            {"exception": ValueError},
        ),
    ],
)
def test_get_baseline_value(
    data: np.ndarray, shape: Tuple, expected: Union[float, dict, bool]
):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            get_baseline_value(
                value=data["value"], return_shape=shape["return_shape"], arr=data["arr"]
            )
        return
    out = get_baseline_value(
        value=data["value"], return_shape=shape["return_shape"], arr=data["arr"]
    )
    assert (
        out == pytest.approx(expected["value"], rel=1e-1, abs=1e-1)
        and out.shape == shape["return_shape"]
    ), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (lazy_fixture("baseline_black_3d"), {"return_shape": (1, 2)}, dict),
        (lazy_fixture("baseline_black_3d"), {"return_shape": (1, 2)}, dict),
        (lazy_fixture("baseline_white_3d"), {"return_shape": (1, 2)}, dict),
        (lazy_fixture("baseline_white_3d"), {"return_shape": (1, 2)}, dict),
        (lazy_fixture("baseline_mean_3d"), {"return_shape": (1, 2)}, dict),
        (lazy_fixture("baseline_mean_3d"), {"return_shape": (1, 2)}, dict),
    ],
)
def test_get_baseline_dict(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = get_baseline_dict(arr=data["arr"], **params)
    print(out, type(out))
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
    out = get_name(name=data)
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
            {"channel_first": False, "softmax": True},
            {"type": TensorFlowModel},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            {"channel_first": False, "softmax": False},
            {"type": TensorFlowModel},
        ),
        (
            lazy_fixture("load_mnist_model"),
            {"channel_first": True, "softmax": True},
            {"type": PyTorchModel},
        ),
        (
            lazy_fixture("load_mnist_model"),
            {"channel_first": True, "softmax": False},
            {"type": PyTorchModel},
        ),
        (
            None,
            {"channel_first": True, "softmax": True},
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
            out = get_wrapped_model(model, **params)
        return
    out = get_wrapped_model(model, **params)
    if "type" in expected:
        isinstance(out, expected["type"]), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {
                "arr_shape": (10, 20, 30, 40),
                "kernel_shape": (3, 3),
                "indices": [15, 22, 30],
                "indexed_axes": [2, 3],
            },
            {"shape": (10, 20, 30, 40)},
        ),
        (
            {
                "arr_shape": (10, 20, 30, 40),
                "kernel_shape": (4, 9, 21),
                "indices": [15, 22, 30],
                "indexed_axes": [0, 1, 2],
            },
            {"shape": (10, 20, 30, 40)},
        ),
        (
            {
                "arr_shape": (10, 20, 30, 40),
                "kernel_shape": (1, 1, 1),
                "indices": [15, 22, 30],
                "indexed_axes": [1, 2, 3],
            },
            {"shape": (10, 20, 30, 40)},
        ),
        (
            {
                "arr_shape": (10, 20, 30, 40),
                "kernel_shape": (1, 3, 3),
                "indices": [15, 22, 30],
                "indexed_axes": [2, 3],
            },
            {"exception": AssertionError},
        ),
    ],
)
def test_blur_at_indices(
    params: dict,
    expected: Union[float, dict, bool],
):
    input = random_array(params["arr_shape"])
    kernel = random_array(params["kernel_shape"])

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = blur_at_indices(
                arr=input,
                kernel=kernel,
                indices=params["indices"],
                indexed_axes=params["indexed_axes"],
            )
        return
    out = blur_at_indices(
        arr=input,
        kernel=kernel,
        indices=params["indices"],
        indexed_axes=params["indexed_axes"],
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
            },
            {"value": (slice(0, 4),)},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0,),
            },
            {"value": (slice(0, 4),)},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0),
            },
            {"value": (slice(0, 4), slice(0, 4))},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0, 0),
            },
            {"value": (slice(0, 4), slice(0, 4))},
        ),
        (
            {
                "patch_size": (4, 6),
                "coords": (0, 0),
            },
            {"value": (slice(0, 4), slice(0, 6))},
        ),
        (
            {
                "patch_size": 10,
                "coords": (1, 2),
            },
            {"value": (slice(1, 11), slice(2, 12))},
        ),
        (
            {
                "patch_size": (10, 5),
                "coords": (1, 2),
            },
            {"value": (slice(1, 11), slice(2, 7))},
        ),
        (
            {
                "patch_size": 4,
                "coords": (0, 0, 0),
            },
            {"value": (slice(0, 4), slice(0, 4), slice(0, 4))},
        ),
        (
            {
                "patch_size": (4, 4, 4),
                "coords": (0, 0),
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0, 0, 0),
            },
            {"exception": ValueError},
        ),
        (
            {
                "patch_size": (4, 4),
                "coords": (0,),
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

    assert np.all(
        out_slice == expected_slice
        for out_slice, expected_slice in zip(out, expected["value"])
    ), f"Slices not equal. {out_slice} != {expected_slice}"


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
                "a_batch": np.ones((64, 1, 1, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 1, 1, 128))},
        ),
        (
            {
                "a_batch": np.ones((64, 3, 128)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 3, 128, 1))},
        ),
        (
            {
                "a_batch": np.ones((64, 3)),
                "x_batch": np.ones((64, 3, 128, 128)),
            },
            {"value": np.ones((64, 3, 1, 1))},
        ),
        (
            {
                "a_batch": np.ones((64, 200)),
                "x_batch": np.ones((64, 3, 128, 200)),
            },
            {"value": np.ones((64, 1, 1, 200))},
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
                "shape": (16,),
            },
            {"value": 4},
        ),
        (
            {
                "patch_size": (4,),
                "shape": (16,),
            },
            {"value": 4},
        ),
        (
            {
                "patch_size": (4, 4),
                "shape": (16,),
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
                "patch_size": (4,),
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


# TODO: Change test cases (and function) for batching update, since currently single images are expected
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
                "a_batch": np.ones((30, 1, 1, 1, 5, 6)),
            },
            {"value": [3, 4]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 1, 1, 5, 6)),
            },
            {"value": [3, 4]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 2, 3, 1, 1, 1)),
            },
            {"value": [0, 1]},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 9, 9, 9, 9, 9, 9)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 2, 2, 2, 2)),
                "a_batch": np.ones((30, 2)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 2)),
                "a_batch": np.ones((30, 2, 2, 2)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 3, 4, 5, 6)),
                "a_batch": np.ones((30, 2, 4, 6)),
            },
            {"exception": ValueError},
        ),
        (
            {
                "x_batch": np.ones((30, 2, 2, 2, 2)),
                "a_batch": np.ones((30, 2, 1, 1, 2)),
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


# TODO: Change test cases (and function) for batching update, since currently single images are expected
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
            {
                "value": (
                    (
                        np.array([[[[[0]]]]]),
                        np.array([[[[[0]]]]]),
                        np.array([[[[[0]]]]]),
                        np.array([[[[[0]]]]]),
                        np.array([[[[[5]]]]]),
                    )
                )
            },
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": [5, 19],
                "indexed_axes": [2, 3, 4],
            },
            {
                "value": (
                    slice(None),
                    slice(None),
                    np.array([[[0, 0]]]),
                    np.array([[[0, 3]]]),
                    np.array([[[5, 1]]]),
                )
            },
        ),
        (
            {
                "arr": np.ones((2, 3, 4, 5, 6)),
                "indices": (np.array([1, 1]), np.array([0, 2])),
                "indexed_axes": [0, 1],
            },
            {"value": (np.array([[1, 1]]), np.array([[0, 2]]))},
        ),
        (
            {
                "arr": np.arange(0, 10000).reshape((10, 10, 10, 10)),
                "indices": (slice(0, 2), slice(0, 3)),
                "indexed_axes": [2, 3],
            },
            {"sum": 2973600, "shape": (10, 10, 2, 3)},
        ),
        (
            {
                "arr": np.arange(0, 10000).reshape((10, 10, 10, 10)),
                "indices": (slice(0, 2), slice(0, 3)),
                "indexed_axes": [0, 1],
            },
            {"sum": 389700, "shape": (2, 3, 10, 10)},
        ),
        (
            {
                "arr": np.arange(0, 10000).reshape((10, 10, 10, 10)),
                "indices": (
                    np.array([0, 1]),
                    np.array([3, 4, 5]),
                    np.array([3, 4, 5, 6]),
                ),
                "indexed_axes": [0, 1, 2],
            },
            {"sum": 227880, "shape": (2, 3, 4, 10)},
        ),
        (
            {
                "arr": np.arange(0, 10000).reshape((10, 10, 10, 10)),
                "indices": (
                    slice(None, None, None),
                    np.array(
                        [
                            [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
                            [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                        ]
                    ),
                    np.array(
                        [
                            [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
                            [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
                        ]
                    ),
                    np.array(
                        [
                            [[3, 8, 7], [3, 8, 7], [3, 8, 7]],
                            [[3, 8, 7], [3, 8, 7], [3, 8, 7]],
                        ]
                    ),
                ),
                "indexed_axes": [1, 2, 3],
            },
            {
                "value": (
                    slice(None, None, None),
                    np.array(
                        [
                            [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
                            [[6, 6, 6], [6, 6, 6], [6, 6, 6]],
                        ]
                    ),
                    np.array(
                        [
                            [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
                            [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
                        ]
                    ),
                    np.array(
                        [
                            [[3, 8, 7], [3, 8, 7], [3, 8, 7]],
                            [[3, 8, 7], [3, 8, 7], [3, 8, 7]],
                        ]
                    ),
                )
            },
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
    else:
        out = expand_indices(**params)
        if "sum" in expected and "shape" in expected:
            assert params["arr"][out].shape == expected["shape"]
            assert params["arr"][out].sum() == expected["sum"]
        else:
            assert all([np.all(a == b) for a, b in list(zip(out, expected["value"]))])


# TODO: Change test cases (and function) for batching update, since currently single images are expected
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
    ],
)
def test_get_leftover_shape(params: dict, expected: Any):
    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = get_leftover_shape(**params)
        return

    out = get_leftover_shape(**params)
    assert all([np.all(a == b) for a, b in list(zip(out, expected["value"]))])
