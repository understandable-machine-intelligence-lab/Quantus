import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.helpers import utils
from quantus.functions.perturb_func import *


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
def input_batch_zeros_2d_3ch_flattened():
    return np.zeros(shape=(2, 3, 224, 224)).reshape(2, -1)


@pytest.fixture
def input_uniform_2d_3ch_flattened():
    return np.random.uniform(0, 0.1, size=(3, 224, 224)).flatten()


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


@pytest.mark.perturb_func
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


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_batch_zeros_2d_3ch_flattened"),
            {
                "indices": np.array([[0, 2], [2, 5]]),
                "perturb_baseline": 1.0,
            },
            1,
        ),
    ],
)
def test_batch_baseline_replacement_by_indices(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    batch_size = data.shape[0]
    # Output
    out = batch_baseline_replacement_by_indices(arr=data, **params)

    # Indices
    indices = params["indices"]

    if isinstance(expected, (int, float)):
        assert np.all(
            out[np.arange(batch_size)[:, None], indices] == expected
        ), f"Test failed.{out}"


@pytest.mark.perturb_func
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


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "coverage,expect_raises",
    [(0.10, False), (0.30, False), (0.50, False), (0.55, True), (0.80, True)],
)
def test_baseline_replacement_by_indices_excluding_feasibility(coverage, expect_raises):
    """The control exists only while the excluded region covers at most half the
    frame: one cannot draw N distinct positions from fewer than N available ones.
    Above the bound the function must RAISE rather than silently perturbing fewer
    positions, because a short draw returns a number that looks like a measurement
    and is not one."""
    side = 100
    arr = np.zeros((3, side, side))
    n_ex = int(coverage * side * side)
    ex = np.zeros(side * side, dtype=bool)
    ex[:n_ex] = True
    ex = ex.reshape(side, side)
    indices = np.arange(n_ex)  # same count as the excluded region
    if expect_raises:
        with pytest.raises(ValueError):
            baseline_replacement_by_indices_excluding(
                arr=arr,
                indices=indices,
                indexed_axes=[1, 2],
                perturb_baseline=1.0,
                exclude_mask=ex,
                seed=0,
            )
    else:
        out = baseline_replacement_by_indices_excluding(
            arr=arr,
            indices=indices,
            indexed_axes=[1, 2],
            perturb_baseline=1.0,
            exclude_mask=ex,
            seed=0,
        )
        assert out.shape == arr.shape


@pytest.mark.perturb_func
def test_baseline_replacement_by_indices_excluding_never_overlaps():
    """No perturbed position may fall inside the excluded region. This is the whole
    point of the function: a control that overlaps the object it is a control for
    destroys, in expectation, a fraction `coverage` of that object."""
    side = 64
    arr = np.zeros((3, side, side))
    ex = np.zeros((side, side), dtype=bool)
    ex[:20, :20] = True
    indices = np.arange(150)
    for seed in range(5):
        out = baseline_replacement_by_indices_excluding(
            arr=arr,
            indices=indices,
            indexed_axes=[1, 2],
            perturb_baseline=1.0,
            exclude_mask=ex,
            seed=seed,
        )
        changed = np.flatnonzero((out != arr).any(axis=0))
        assert len(changed) == len(indices)
        assert not ex.reshape(-1)[changed].any()


@pytest.mark.perturb_func
def test_baseline_replacement_by_indices_excluding_matches_stock_when_mask_is_none():
    """With no excluded region the function must be the stock behaviour, so it is a
    drop-in and cannot change existing results."""
    arr = np.zeros((3, 32, 32))
    indices = np.arange(40)
    a = baseline_replacement_by_indices(
        arr=arr, indices=indices, indexed_axes=[1, 2], perturb_baseline=1.0
    )
    b = baseline_replacement_by_indices_excluding(
        arr=arr,
        indices=indices,
        indexed_axes=[1, 2],
        perturb_baseline=1.0,
        exclude_mask=None,
    )
    assert np.array_equal(a, b)


@pytest.mark.perturb_func
def test_baseline_replacement_by_indices_excluding_is_seeded():
    """Same seed, same draw; different seed, different draw. A control that cannot
    be reproduced cannot be audited."""
    arr = np.zeros((3, 32, 32))
    ex = np.zeros((32, 32), dtype=bool)
    ex[:8] = True
    kw = dict(
        arr=arr,
        indices=np.arange(30),
        indexed_axes=[1, 2],
        perturb_baseline=1.0,
        exclude_mask=ex,
    )
    p = baseline_replacement_by_indices_excluding(seed=7, **kw)
    q = baseline_replacement_by_indices_excluding(seed=7, **kw)
    r = baseline_replacement_by_indices_excluding(seed=8, **kw)
    assert np.array_equal(p, q)
    assert not np.array_equal(p, r)
