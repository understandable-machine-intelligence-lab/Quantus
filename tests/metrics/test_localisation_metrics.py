from typing import Union

import numpy as np
from PIL import Image
from glob import glob
from tensorflow.keras.datasets import cifar10
from torchvision.models.resnet import resnet18
from torchvision import transforms
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain


@pytest.fixture
def all_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_no_abatch_1d_1ch():
    s_batch = np.zeros((10, 1, 28))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 28))
    s_batch[:, :, 0:15] = 1.0
    return {
        "x_batch": np.random.randn(10, 1, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_no_abatch_2d_1ch():
    s_batch = np.zeros((10, 1, 28, 28))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 28, 28))
    s_batch[:, :, 0:15, 0:15] = 1.0
    return {
        "x_batch": np.random.randn(10, 1, 28, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_zeros_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_non_normalised_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 20, size=(10, 1, 224))
    s_batch[:, :, 50:150] = 1.0
    a_batch[:, :, 50:150] = 25
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_non_normalised_2d():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 20, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 25
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_seg_bigger_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 0:150] = 1.0
    a_batch[:, :, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def all_in_gt_seg_bigger_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:150, 0:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_zeros_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_fourth_1d():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:112] = 1.0
    a_batch[:, :, 112:224] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def none_in_gt_fourth_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:112, 0:112] = 1.0
    a_batch[:, :, 112:224, 112:224] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_zeros_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 50:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_zeros_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224))
    s_batch[:, :, 50:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def half_in_gt_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


# TODO: unused fixture
@pytest.fixture
def half_in_gt_zeros_bigger_1d_3ch():
    s_batch = np.zeros((10, 1, 224))
    a_batch = np.zeros((10, 1, 224))
    s_batch[:, :, 0:100] = 1.0
    a_batch[:, :, 0:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


# TODO: unused fixture
@pytest.fixture
def half_in_gt_zeros_bigger_2d_3ch():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
        "s_batch": s_batch,
    }


@pytest.fixture
def load_artificial_attribution():
    """Build an artificial attribution map"""
    zeros = np.zeros((1, 28, 28))
    ones = np.ones((1, 28, 28))
    mosaics_list = []
    images = [zeros, ones]
    indices_list = [
        tuple([0, 0, 1, 1]),
        tuple([1, 1, 0, 0]),
        tuple([0, 1, 0, 1]),
        tuple([1, 0, 1, 0]),
    ]
    for indices in indices_list:
        first_row = np.concatenate((images[indices[0]], images[indices[1]]), axis=1)
        second_row = np.concatenate((images[indices[2]], images[indices[3]]), axis=1)
        mosaic = np.concatenate((first_row, second_row), axis=2)
        mosaics_list.append(mosaic)
    return np.array(mosaics_list)


@pytest.fixture()
def load_mnist_adaptive_lenet_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetAdaptivePooling(input_shape=(1, 28, 28))
    model.load_state_dict(
        torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def load_mnist_mosaics():
    """Load a batch of MNIST digits and build mosaics from them"""
    x_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_y"), dtype=torch.int64
    ).numpy()
    mosaics_returns = mosaic_creation(
        images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777
    )
    (
        all_mosaics,
        mosaic_indices_list,
        mosaic_labels_list,
        p_batch_list,
        target_list,
    ) = mosaics_returns
    return {
        "x_batch": all_mosaics,
        "y_batch": target_list,
        "custom_batch": p_batch_list,
    }


@pytest.fixture()
def load_cifar10_adaptive_lenet_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetAdaptivePooling(input_shape=(3, 32, 32))
    model.load_state_dict(
        torch.load("tests/assets/cifar10", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def load_cifar10_mosaics():
    """Load a batch of Cifar10 and build mosaics from them"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_batch = torch.as_tensor(
        x_train[:124, ...].reshape(124, 3, 32, 32),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(y_train[:124].reshape(124), dtype=torch.int64).numpy()
    mosaics_returns = mosaic_creation(
        images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777
    )
    (
        all_mosaics,
        mosaic_indices_list,
        mosaic_labels_list,
        p_batch_list,
        target_list,
    ) = mosaics_returns
    return {
        "x_batch": all_mosaics,
        "y_batch": target_list,
        "custom_batch": p_batch_list,
    }


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            True,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            False,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            False,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            True,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            True,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            True,
        ),
    ],
)
def test_pointing_game(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = PointingGame(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )
    if isinstance(expected, bool):
        assert all(s == expected for s in scores), "Test failed."
    elif isinstance(expected, dict):
        assert isinstance(scores, expected["type"]), "Test failed."
    elif isinstance(expected, list):
        assert all(s == e for s, e in zip(scores, expected)), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "k": 100,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "k": 10000,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "k": 200,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.5,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "k": 40000,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.25,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "k": 20,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "k": 500,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "init": {
                    "k": 100,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "init": {
                    "k": 10000,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "k": 200,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.38,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "k": 40000,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.1, "max": 0.25},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "k": 50,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.9800000000000001,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "k": 2500,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "k": 125,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.4,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "k": 1250,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "k": 100,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "k": 10000,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "k": 100,
                    "concept_influence": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            2.24,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "k": 10000,
                    "concept_influence": True,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            5.0176,  # TODO: verify correctness
        ),
    ],
)
def test_top_k_intersection(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = TopKIntersection(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )

    if isinstance(expected, float):
        assert all(round(s, 2) == round(expected, 2) for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_seg_bigger_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_seg_bigger_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.5,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
    ],
)
def test_relevance_mass_accuracy(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = RelevanceMassAccuracy(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )

    print(scores)

    if isinstance(expected, float):
        assert (
            all(round(s, 2) == round(expected, 2) for s in scores) == True
        ), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores) == True, "Test failed."
        assert all(s < expected["max"] for s in scores) == True, "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_seg_bigger_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_seg_bigger_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_1d_3ch"),
            {
                "init": {
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_2d_3ch"),
            {
                "init": {
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("half_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.5, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("half_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.5,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
    ],
)
def test_relevance_rank_accuracy(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = RelevanceRankAccuracy(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )

    """
    if "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."
    """
    if isinstance(expected, float):
        print(scores)
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_non_normalised_1d_3ch"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_non_normalised_2d"),
            {
                "init": {
                    "normalise": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_fourth_1d"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_fourth_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.33333333333333337,  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
    ],
)
def test_auc(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = AUC(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), f"Test failed. {scores[0]}"
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_no_abatch_1d_1ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_no_abatch_2d_1ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                },
            },
            {"type": list},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_1d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.8, "max": 0.95},  # TODO: verify correctness
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_2d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            {"min": 0.8, "max": 0.85},
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("none_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "weighted": True,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("none_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "weighted": True,
                    "abs": False,
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
            },
            0.0,
        ),
        (
            lazy_fixture("load_1d_1ch_conv_model"),
            lazy_fixture("all_in_gt_zeros_1d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("all_in_gt_zeros_2d_3ch"),
            {
                "init": {
                    "weighted": False,
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
            },
            1.0,
        ),
    ],
)
def test_attribution_localisation(
    model,
    data: dict,
    params: dict,
    expected: Union[bool, dict],
):
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    scores = AttributionLocalisation(**init_params)(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        a_batch=data["a_batch"],
        s_batch=data["s_batch"],
        **call_params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    elif "type" in expected:
        assert isinstance(scores, expected["type"]), "Test failed."
    else:
        print(scores)
        assert all(s > expected["min"] for s in scores), "Test failed."
        assert all(s < expected["max"] for s in scores), "Test failed."


@pytest.mark.localisation
@pytest.mark.parametrize(
    "model,mosaic_data,a_batch,params,expected",
    [
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradientShap",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "normalise": True,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "IntegratedGradients",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "InputXGradient",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Saliency",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradCam",
                        "gc_layer": "model._modules.get('conv_2')",
                        "interpolate": (56, 56),
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_cifar10_adaptive_lenet_model"),
            lazy_fixture("load_cifar10_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "GradCam",
                        "gc_layer": "model._modules.get('conv_2')",
                        "interpolate": (64, 64),
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            lazy_fixture("load_mnist_mosaics"),
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Control Var. Sobel Filter",
                    },
                },
            },
            None,
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            {
                "x_batch": None,
                "y_batch": None,
                "custom_batch": None,
            },
            None,
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {
                    "explain_func": explain,
                    "explain_func_kwargs": {
                        "method": "Gradient",
                    },
                },
            },
            {"exception": ValueError},
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            {
                "x_batch": np.ones((4, 1, 56, 56)),
                "y_batch": np.ones(4),
                "custom_batch": [
                    tuple([0, 0, 1, 1]),
                    tuple([1, 1, 0, 0]),
                    tuple([0, 1, 0, 1]),
                    tuple([1, 0, 1, 0]),
                ],
            },
            lazy_fixture("load_artificial_attribution"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"value": 1},
        ),
        (
            lazy_fixture("load_mnist_adaptive_lenet_model"),
            {
                "x_batch": np.ones((4, 1, 56, 56)),
                "y_batch": np.ones(4),
                "custom_batch": [
                    tuple([1, 1, 0, 0]),
                    tuple([0, 0, 1, 1]),
                    tuple([1, 0, 1, 0]),
                    tuple([0, 1, 0, 1]),
                ],
            },
            lazy_fixture("load_artificial_attribution"),
            {
                "init": {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                "call": {},
            },
            {"value": 0},
        ),
    ],
)
def test_focus(
    model: Optional[ModelInterface],
    mosaic_data: Dict[str, Union[np.ndarray, list]],
    a_batch: Optional[np.ndarray],
    params: dict,
    expected: Optional[dict],
):
    x_batch, y_batch, custom_batch = (
        mosaic_data["x_batch"],
        mosaic_data["y_batch"],
        mosaic_data["custom_batch"],
    )
    init_params = params.get("init", {})
    call_params = params.get("call", {})

    metric = Focus(**init_params)

    if expected and "exception" in expected:
        with pytest.raises(expected["exception"]):
            metric(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                custom_batch=custom_batch,
                **call_params,
            )
        return

    custom_batch_len = len(custom_batch)
    while len(custom_batch) > 0:
        if x_batch is not None:
            x_batch_mini, x_batch = x_batch[:10], x_batch[10:]
        else:
            x_batch_mini = None
        if y_batch is not None:
            y_batch_mini, y_batch = y_batch[:10], y_batch[10:]
        else:
            y_batch_mini = None
        if a_batch is not None:
            a_batch_mini, a_batch = a_batch[:10], a_batch[10:]
        else:
            a_batch_mini = None

        custom_batch_mini, custom_batch = custom_batch[:10], custom_batch[10:]

        scores = metric(
            model=model,
            x_batch=x_batch_mini,
            y_batch=y_batch_mini,
            a_batch=a_batch_mini,
            custom_batch=custom_batch_mini,
            **call_params,
        )

    assert len(scores) == len(custom_batch_mini), "Test failed."
    assert all([0 <= score <= 1 for score in scores]), "Test failed."
    if expected and "value" in expected:
        assert all((score == expected["value"]) for score in scores), "Test failed."
