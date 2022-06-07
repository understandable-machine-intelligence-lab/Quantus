from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers import perturb_func
from ...quantus.helpers.explanation_func import explain


@pytest.fixture()
def load_mnist_adaptive_lenet_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetAdaptivePooling()
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def load_mnist_mosaics():
    """Load a batch of MNIST digits and build mosaics from them"""
    x_batch = torch.as_tensor(
        np.loadtxt("tutorials/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        np.loadtxt("tutorials/assets/mnist_y"), dtype=torch.int64
    ).numpy()
    mosaics_returns = mosaic_creation(images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777)
    all_mosaics, mosaic_indices_list, mosaic_labels_list, p_batch_list, target_list = mosaics_returns

    return {
        "all_mosaics": all_mosaics,
        "mosaic_indices_list": mosaic_indices_list,
        "mosaic_labels_list": mosaic_labels_list,
        "p_batch_list": p_batch_list,
        "target_list": target_list,
    }


@pytest.mark.confusion
@pytest.mark.parametrize(
    "model,data,a_batch,params",
    [
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
        ),
    ],
)
def test_focus(
        model,
        data: np.ndarray,
        a_batch: Optional[np.ndarray],
        params: dict,
):
    x_batch, y_batch, p_batch = (
        data["all_mosaics"],
        data["target_list"],
        data["p_batch_list"]
    )
    metric = Focus(**params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        p_batch=p_batch,
        **params,
    )


