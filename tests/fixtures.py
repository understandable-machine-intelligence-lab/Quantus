from typing import Tuple
import os
import pytest
import pickle
import torch
import torchvision
from torchvision import transforms
import numpy as np
from ..quantus.helpers.models import LeNet


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = torch.as_tensor(
        np.loadtxt("tutorials/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    )
    y_batch = torch.as_tensor(np.loadtxt("tutorials/assets/mnist_y"), dtype=torch.int64)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture
def almost_uniform(scope="session", autouse=True):
    a_batch = np.random.uniform(0, 0.01, size=(10, 1, 224, 224))
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
    }
