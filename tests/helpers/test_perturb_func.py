from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *
from ...quantus.helpers import utils


@pytest.fixture
def input_zeros_1d_1ch():
    return np.zeros(shape=(1, 224))


@pytest.fixture
def input_zeros_1d_3ch():
    return np.zeros(shape=(3, 224))


@pytest.fixture
def input_zeros_2d_1ch():
    return np.zeros(shape=(1, 224, 224))


@pytest.fixture
def input_zeros_2d_3ch():
    return np.zeros(shape=(3, 224, 224))


@pytest.fixture
def input_zeros_2d_3ch_flattened():
    return np.zeros(shape=(3, 224, 224)).flatten()


@pytest.fixture
def input_uniform_2d_3ch_flattened():
    return np.random.uniform(0, 0.1, size=(3, 224, 224)).flatten()


@pytest.fixture
def input_ones_mnist():
    return np.ones(shape=(1, 28, 28))


@pytest.fixture
def input_ones_mnist_flattened():
    return np.ones(shape=(1, 28, 28)).flatten()


@pytest.fixture
def input_zeros_mnist_flattened():
    return np.zeros(shape=(1, 28, 28)).flatten()


@pytest.fixture
def input_uniform_1d_3ch():
    return np.random.uniform(0, 0.1, size=(3, 224))


@pytest.fixture
def input_uniform_2d_3ch():
    return np.random.uniform(0, 0.1, size=(3, 224, 224))


@pytest.fixture
def input_uniform_2d_3ch_flattened():
    return np.random.uniform(0, 0.1, size=(3, 224, 224)).flatten()


@pytest.fixture
def input_uniform_3d_3ch():
    return np.random.uniform(0, 0.1, size=(3, 224, 224, 224))


@pytest.fixture
def input_uniform_mnist():
    return np.random.uniform(0, 0.1, size=(1, 28, 28))


@pytest.mark.fixed
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [1, 2],
                "perturb_baseline": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [0],
                "perturb_baseline": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [0, 1, 2],
                "perturb_baseline": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch_flattened"),
            {
                "indices": [0, 2],
                "indexed_axes": [0],
                "perturb_baseline": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_zeros_1d_1ch"),
            {
                "indices": [0, 2, 112, 113, 128, 223],
                "indexed_axes": [0, 1],
                "perturb_baseline": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_zeros_1d_3ch"),
            {
                "indices": [0, 2, 112, 113, 128, 223],
                "indexed_axes": [1],
                "perturb_baseline": np.array([1, 2, 3]),
            },
            np.array([1, 2, 3]),
        ),
        (
            lazy_fixture("input_zeros_2d_1ch"),
            {
                "indices": [0, 2, 224, 226, 448, 450],
                "indexed_axes": [1, 2],
                "perturb_baseline": np.array([1]),
            },
            np.array([1]),
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2, 224, 226, 448, 450],
                "indexed_axes": [0, 1, 2],
                "perturb_baseline": 1.0,
            },
            1,
        ),
    ],
)
def test_baseline_replacement_by_indices(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    # Output
    out = baseline_replacement_by_indices(arr=data, **params)

    # Indices
    indices = np.unravel_index(
        params["indices"], tuple([data.shape[i] for i in params["indexed_axes"]])
    )
    if not np.array(indices).ndim > 1:
        indices = [np.array(indices)]
    indices = list(indices)
    for i in range(0, params["indexed_axes"][0]):
        indices = slice(None), *indices
    indices = tuple(indices)

    if isinstance(expected, (int, float)):
        assert np.all(
            [i == expected for i in out[indices].flatten()]
        ), f"Test failed.{out}"


@pytest.mark.fixed
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [1, 2],
                "input_shift": -1.0,
            },
            -1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [0],
                "input_shift": -1.0,
            },
            -1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2],
                "indexed_axes": [0, 1, 2],
                "input_shift": -1.0,
            },
            -1,
        ),
        (
            lazy_fixture("input_zeros_2d_3ch_flattened"),
            {
                "indices": [0, 2],
                "indexed_axes": [0],
                "input_shift": -1.0,
            },
            -1,
        ),
        (
            lazy_fixture("input_zeros_1d_1ch"),
            {
                "indices": [0, 2, 112, 113, 128, 223],
                "indexed_axes": [0, 1],
                "input_shift": -1.0,
            },
            -1,
        ),
        (
            lazy_fixture("input_zeros_1d_3ch"),
            {
                "indices": [0, 2, 112, 113, 128, 223],
                "indexed_axes": [1],
                "input_shift": np.array([1, 2, 3]),
            },
            np.array([1, 2, 3]),
        ),
        (
            lazy_fixture("input_zeros_2d_1ch"),
            {
                "indices": [0, 2, 224, 226, 448, 450],
                "indexed_axes": [1, 2],
                "input_shift": np.array([1]),
            },
            np.array([1]),
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "indices": [0, 2, 224, 226, 448, 450],
                "indexed_axes": [0, 1, 2],
                "input_shift": 1.0,
            },
            1,
        ),
    ],
)
def test_baseline_replacement_by_shift(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    # Output
    out = baseline_replacement_by_shift(arr=data, **params)

    # Indices
    indices = np.unravel_index(
        params["indices"], tuple([data.shape[i] for i in params["indexed_axes"]])
    )
    if not np.array(indices).ndim > 1:
        indices = [np.array(indices)]
    indices = list(indices)
    for i in range(0, params["indexed_axes"][0]):
        indices = slice(None), *indices
    indices = tuple(indices)

    if isinstance(expected, (int, float)):
        assert np.all(
            [i == expected for i in out[indices].flatten()]
        ), f"Test failed.{out}"


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "blur_kernel_size": 15,
                "indices": [20, 15, 5, 27, 9],
                "indexed_axes": [0, 1, 2],
            },
            {},
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "blur_kernel_size": [3, 4],
                "indices": [20, 15, 5, 27, 9],
                "indexed_axes": [1, 2],
            },
            {},
        ),
        (
            lazy_fixture("input_uniform_mnist"),
            {
                "blur_kernel_size": 15,
                "indices": [20, 15, 5, 27, 9],
                "indexed_axes": [0, 1],
            },
            {},
        ),
        (
            lazy_fixture("input_uniform_1d_3ch"),
            {
                "blur_kernel_size": 15,
                "indices": [20, 15, 5, 27, 9],
                "indexed_axes": [1],
            },
            {},
        ),
        (
            lazy_fixture("input_uniform_3d_3ch"),
            {
                "blur_kernel_size": 15,
                "indices": [20, 15, 5, 27, 9],
                "indexed_axes": [0, 1, 2, 3],
            },
            {},
        ),
    ],
)
def test_baseline_replacement_by_blur(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = baseline_replacement_by_blur(
                arr=data,
                indices=params["indices"],
                indexed_axes=params["indexed_axes"],
                blur_kernel_size=params["blur_kernel_size"],
            )
        return

    out = baseline_replacement_by_blur(
        arr=data,
        indices=params["indices"],
        indexed_axes=params["indexed_axes"],
        blur_kernel_size=params["blur_kernel_size"],
    )

    indices = utils.expand_indices(data, params["indices"], params["indexed_axes"])
    patch_mask = np.zeros(data.shape, dtype=bool)
    patch_mask[indices] = True
    assert out.shape == data.shape, "Test failed."
    assert np.all(out[patch_mask] != data[patch_mask]), "Test failed."
    assert np.all(out[~patch_mask] == data[~patch_mask]), "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_1d_3ch"),
            {
                "indices": [0],
                "indexed_axes": [0, 1],
            },
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "indices": [0],
                "indexed_axes": [0, 1, 2],
            },
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch_flattened"),
            {
                "indices": [0],
                "indexed_axes": [0],
            },
            True,
        ),
    ],
)
def test_gaussian_noise(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = gaussian_noise(arr=data, **params)
    assert any(out.flatten()[0] != out.flatten()), "Test failed."
    assert any(out.flatten() != data.flatten()) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_1d_3ch"),
            {
                "perturb_radius": 0.02,
                "indices": [0],
                "indexed_axes": [0, 1],
            },
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "perturb_radius": 0.02,
                "indices": [0],
                "indexed_axes": [0, 1],
            },
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch_flattened"),
            {
                "perturb_radius": 0.02,
                "indices": [0],
                "indexed_axes": [0],
            },
            True,
        ),
    ],
)
def test_uniform_noise(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = uniform_noise(arr=data, **params)
    assert any(out.flatten()[0] != out.flatten()), "Test failed."
    assert any(out.flatten() != data.flatten()) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_angle": 30},
            True,
        ),
    ],
)
def test_rotation(data: dict, params: dict, expected: Union[float, dict, bool]):
    out = rotation(arr=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_dx": 20, "perturb_baseline": "black"},
            True,
        )
    ],
)
def test_translation_x_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_x_direction(arr=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_dx": 20, "perturb_baseline": "black"},
            True,
        )
    ],
)
def test_translation_y_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_y_direction(arr=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_dx": 20},
            True,
        ),
    ],
)
def test_no_perturbation(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = no_perturbation(arr=data, **params)
    assert (out == data).all() == expected, "Test failed."
