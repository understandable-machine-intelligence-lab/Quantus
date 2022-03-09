import pytest
from typing import Union
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
def input_ones_mnist_flattened():
    return np.ones(shape=(1, 28, 28)).flatten()


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
def input_uniform_mnist():
    return np.random.uniform(0, 0.1, size=(1, 28, 28))


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_1d_3ch"),
            {},
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {},
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch_flattened"),
            {},
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
            lazy_fixture("input_zeros_2d_3ch_flattened"),
            {
                "indices": [0, 2],
                "fixed_values": 1.0,
            },
            1,
        ),
        (
            lazy_fixture("input_ones_mnist_flattened"),
            {
                "indices": np.arange(0, 784),
                "input_shift": -1.0,
            },
            -1,
        ),
    ],
)
def test_baseline_replacement_by_indices(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = baseline_replacement_by_indices(arr=data, **params)

    if isinstance(expected, (int, float)):
        assert np.all([i == expected for i in out[params["indices"]]]), "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_zeros_1d_1ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (0,),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_1d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (0,),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_2d_1ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (0, 0),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (0, 0),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 10,
                "coords": (0, 0),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (11, 22),
            },
            {},
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (11, ),
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("input_zeros_2d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (11, 11, 11, ),
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("input_zeros_1d_3ch"),
            {
                "perturb_baseline": 1.0,
                "patch_size": 4,
                "coords": (11, 11, ),
            },
            {"exception": ValueError},
        ),
    ],
)
def test_baseline_replacement_by_patch(
        data: np.ndarray, params: dict, expected: dict
):
    patch_slice = utils.create_patch_slice(
        patch_size=params["patch_size"],
        coords=params["coords"],
        expand_first_dim=True,
    )

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            out = baseline_replacement_by_patch(
                arr=data,
                patch_slice=patch_slice,
                perturb_baseline=params["perturb_baseline"],
            )
        return

    out = baseline_replacement_by_patch(
        arr=data,
        patch_slice=patch_slice,
        perturb_baseline=params["perturb_baseline"],
    )

    patch_mask = np.zeros(data.shape, dtype=bool)
    patch_mask[patch_slice] = True
    assert np.all(out[patch_mask] != data[patch_mask]), "Test failed."
    assert np.all(out[~patch_mask] == data[~patch_mask]), "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_1d_3ch"),
            {"perturb_radius": 0.02},
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_radius": 0.02},
            True,
        ),
        (
            lazy_fixture("input_uniform_2d_3ch_flattened"),
            {"perturb_radius": 0.02},
            True,
        ),
    ],
)
def test_uniform_sampling(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = uniform_sampling(arr=data, **params)
    assert any(out.flatten()[0] != out.flatten()), "Test failed."
    assert any(out.flatten() != data.flatten()) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("input_uniform_2d_3ch"), {"perturb_angle": 30, "img_size": 224}, True)],
)
def test_rotation(data: dict, params: dict, expected: Union[float, dict, bool]):
    out = rotation(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_dx": 20, "perturb_baseline": "black", "img_size": 224},
            True,
        )
    ],
)
def test_translation_x_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_x_direction(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {"perturb_dx": 20, "perturb_baseline": "black", "img_size": 224},
            True,
        )
    ],
)
def test_translation_y_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_y_direction(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected", [(lazy_fixture("input_uniform_2d_3ch"), {"perturb_dx": 20}, True)]
)
def test_no_perturbation(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = no_perturbation(img=data, **params)
    print(out == data)
    assert (out == data).all() == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "nr_channels": 3,
                "img_size": 224,
                "blur_patch_size": 15,
                "patch_size": 4,
                "top_left_y": 0,
                "top_left_x": 0,
            },
            {"shape": True, "values": False},
        ),
        (
            lazy_fixture("input_uniform_2d_3ch"),
            {
                "nr_channels": 3,
                "img_size": 224,
                "blur_patch_size": 7,
                "patch_size": 4,
                "top_left_y": 0,
                "top_left_x": 0,
            },
            {"shape": True, "values": False},
        ),
        (
            lazy_fixture("input_uniform_mnist"),
            {
                "nr_channels": 1,
                "img_size": 28,
                "blur_patch_size": 15,
                "patch_size": 4,
                "top_left_y": 0,
                "top_left_x": 0,
            },
            {"shape": True, "values": False},
        ),
    ],
)
def test_baseline_replacement_by_blur(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = baseline_replacement_by_blur(img=data, **params)
    assert (out.shape == data.shape) == expected["shape"], "Test failed."
    assert (out == data).all() == expected["values"], "Test failed."
