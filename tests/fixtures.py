from typing import Tuple
import os
import pytest
import pickle
import torch
import torchvision
from torchvision import transforms
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

from ..quantus.helpers.models import LeNet, LeNetTF, ConvNet1D, ConvNet1DTF
from ..quantus.helpers.pytorch_model import PyTorchModel
from ..quantus.helpers.tf_model import TensorFlowModel


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


# @pytest.fixture(scope="session", autouse=True)
# def load_cifar10_model():
#    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
#   model = LeNet(nr_channels=3)
#    model.load_state_dict(
#        torch.load("tests/assets/cifar10", map_location="cpu", pickle_module=pickle)
#    )
#    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model_tf():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetTF()
    model.load_weights("tests/assets/mnist_tf_weights/")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_1ch_conv_model():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1D(n_channels=1, n_classes=10)
    model.eval()
    # TODO: add trained model weights
    # model.load_state_dict(
    #    torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    # )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_3ch_conv_model():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1D(n_channels=3, n_classes=10)
    model.eval()
    # TODO: add trained model weights
    # model.load_state_dict(
    #    torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    # )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_3ch_conv_model_tf():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1DTF(n_channels=3, seq_len=100, n_classes=10)
    # TODO: add trained model weights
    # model = LeNetTF()
    # model.load_weights("tests/assets/mnist_tf_weights/")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_y"), dtype=torch.int64
    ).numpy()
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_cifar10_images():
    """Load a batch of Cifar10 digits: inputs and outputs to use for testing."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_batch = torch.as_tensor(
        x_train[:124, ...].reshape(124, 3, 32, 32),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(y_train[:124].reshape(124), dtype=torch.int64).numpy()
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_tf():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        np.loadtxt("tests/assets/mnist_y"), dtype=torch.int64
    ).numpy()
    return {"x_batch": np.moveaxis(x_batch, 1, -1), "y_batch": y_batch}


@pytest.fixture
def almost_uniform_1d(scope="session", autouse=True):
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 100)),
    }


@pytest.fixture
def almost_uniform_1d_no_abatch_channel_last(scope="session", autouse=True):
    return {
        "x_batch": np.random.randn(10, 100, 3),
        "y_batch": np.random.randint(0, 10, size=10),
    }


@pytest.fixture
def almost_uniform_1d_no_abatch(scope="session", autouse=True):
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture
def almost_uniform_2d(scope="session", autouse=True):
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 224, 224)),
    }


@pytest.fixture
def almost_uniform_2d_no_abatch(scope="session", autouse=True):
    return {
        "x_batch": np.random.randn(10, 1, 28, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture
def flat_image_array(scope="session", autouse=True):
    return {
        "x": np.zeros((1, 3 * 28 * 28)),
        "shape": (3, 28, 28),
        "channel_first": True,
    }


@pytest.fixture
def flat_sequence_array(scope="session", autouse=True):
    return {
        "x": np.zeros((1, 3 * 28)),
        "shape": (3, 28),
        "channel_first": True,
    }
