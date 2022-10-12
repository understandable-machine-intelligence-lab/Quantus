from typing import Union

import numpy as np
import pytest
import torch
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def atts_normalise_seq_0():
    return np.array([0.0, 0.0])


@pytest.fixture
def atts_normalise_seq_1():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, -1.0])


@pytest.fixture
def atts_normalise_seq_2():
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def atts_normalise_seq_3():
    return np.array([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])


@pytest.fixture
def atts_normalise_seq_with_batch_dim():
    return np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
        ]
    )


@pytest.fixture
def atts_normalise_img_with_batch_dim():
    return np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            [
                [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            ],
        ]
    )


@pytest.fixture
def atts_denormalise():
    return np.zeros((3, 2, 2))


@pytest.fixture
def atts_denormalise_torch():
    return torch.tensor(np.zeros((3, 2, 2)))


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_normalise_seq_0"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_1"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, -0.2]),
        ),
        (
            lazy_fixture("atts_normalise_seq_2"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_3"),
            {"normalized_axes": [0]},
            np.array([0.0, -0.2, -0.4, -0.6, -0.8, -1.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_with_batch_dim"),
            {"normalized_axes": [1]},
            np.array(
                [
                    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                ]
            ),
        ),
        (
            lazy_fixture("atts_normalise_img_with_batch_dim"),
            {"normalized_axes": [1, 2]},
            np.array(
                [
                    [
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ],
                    [
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                    ],
                ]
            ),
        ),
        (
            lazy_fixture("atts_normalise_seq_1"),
            {"normalized_axes": None},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, -0.2]),
        ),
    ],
)
def test_normalise_by_max(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = normalise_by_max(a=data, **params)
    assert np.all(out == expected), f"Test failed. (expected: {expected}, is: {out})"


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_normalise_seq_0"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_1"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, -1.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_2"),
            {"normalized_axes": [0]},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_3"),
            {"normalized_axes": [0]},
            np.array([0.0, -0.2, -0.4, -0.6, -0.8, -1.0]),
        ),
        (
            lazy_fixture("atts_normalise_seq_with_batch_dim"),
            {"normalized_axes": [1]},
            np.array(
                [
                    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                ]
            ),
        ),
        (
            lazy_fixture("atts_normalise_img_with_batch_dim"),
            {"normalized_axes": [1, 2]},
            np.array(
                [
                    [
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ],
                    [
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                        [0.0, -0.2, -0.4, -0.6, -0.8, -1.0],
                    ],
                ]
            ),
        ),
        (
            lazy_fixture("atts_normalise_seq_2"),
            {"normalized_axes": None},
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
    ],
)
def test_normalise_by_negative(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = normalise_by_negative(a=data, **params)
    assert np.all(out == expected), f"Test failed. (expected: {expected}, is: {out})"


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_denormalise"),
            {},
            np.array(
                [
                    [[0.485, 0.485], [0.485, 0.485]],
                    [[0.456, 0.456], [0.456, 0.456]],
                    [[0.406, 0.406], [0.406, 0.406]],
                ]
            ),
        ),
        (
            [1, 2],
            {},
            [1, 2],
        ),
    ],
)
def test_denormalise(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = denormalise(
        a=data,
        mean=np.array([0.485, 0.456, 0.406]),
        std=np.array([0.229, 0.224, 0.225]),
        **params,
    )
    assert np.all(
        o == e for o, e in zip(np.array(out).flatten(), np.array(expected).flatten())
    ), "Test failed."
