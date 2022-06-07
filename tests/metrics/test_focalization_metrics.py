from typing import Union

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers import perturb_func
from ...quantus.helpers.explanation_func import explain


@pytest.mark.focalisation
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
        ),
    ],
)


def test_focus(
        model,
        data: np.ndarray,
        params: dict,
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "p_batch" in data:
        p_batch = data["p_batch"]
    else:
        p_batch = None

    metric = Focus(**params)

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        p_batch=p_batch,
        **params,
    )


