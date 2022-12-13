import pytest
import pickle
import torch
import numpy as np
from keras.datasets import cifar10

from quantus.helpers.model.models import LeNet, LeNetTF, ConvNet1D, ConvNet1DTF

CIFAR_IMAGE_SIZE = 32
MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model_tf():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetTF()
    model.load_weights("tests/assets/lenet_mnist_weights.keras")
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
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_cifar10_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_batch = (
        x_train[:BATCH_SIZE]
        .reshape((BATCH_SIZE, 3, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE))
        .astype(float)
    )[:MINI_BATCH_SIZE]
    y_batch = y_train[:BATCH_SIZE].reshape(-1).astype(int)[:MINI_BATCH_SIZE]
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_tf(load_mnist_images):
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""

    return {
        "x_batch": np.moveaxis(load_mnist_images["x_batch"], 1, -1),
        "y_batch": load_mnist_images["y_batch"],
    }


@pytest.fixture(scope="session", autouse=True)
def almost_uniform_1d():
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 100)),
    }


@pytest.fixture(scope="session", autouse=True)
def almost_uniform_1d_no_abatch_channel_last():
    return {
        "x_batch": np.random.randn(10, 100, 3),
        "y_batch": np.random.randint(0, 10, size=10),
    }


@pytest.fixture(scope="session", autouse=True)
def almost_uniform_1d_no_abatch():
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture(scope="session", autouse=True)
def almost_uniform_2d():
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 224, 224)),
    }


@pytest.fixture(scope="session", autouse=True)
def almost_uniform_2d_no_abatch():
    return {
        "x_batch": np.random.randn(10, 1, 28, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture(scope="session", autouse=True)
def flat_image_array():
    return {
        "x": np.zeros((1, 3 * 28 * 28)),
        "shape": (3, 28, 28),
        "channel_first": True,
    }


@pytest.fixture(scope="session", autouse=True)
def flat_sequence_array():
    return {
        "x": np.zeros((1, 3 * 28)),
        "shape": (3, 28),
        "channel_first": True,
    }
