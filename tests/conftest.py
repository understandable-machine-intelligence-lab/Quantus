import pytest
import pickle
import torch
import numpy as np
import tensorflow as tf


from ..quantus.helpers.models import LeNet, LeNetTF, ConvNet1D, ConvNet1DTF, CNN_2D_TF


CIFAR_IMAGE_SIZE = 32
MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    model.training = False
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model_tf():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetTF()
    model.load_weights("tests/weights/lenet_mnist_weights.keras")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_1ch_conv_model():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1D(n_channels=1, n_classes=10)
    model.eval()
    # TODO: add trained model weights
    # model.load_state_dict(
    #    torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    # )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_3ch_conv_model():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1D(n_channels=3, n_classes=10)
    model.eval()
    # TODO: add trained model weights
    # model.load_state_dict(
    #    torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    # )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_1d_3ch_conv_model_tf():
    """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
    model = ConvNet1DTF(n_channels=3, seq_len=100, n_classes=10)
    # TODO: add trained model weights
    # model = LeNetTF()
    # model.load_weights("tutorials/assets/mnist_tf_weights/")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_batch = x_train[:BATCH_SIZE].reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)).astype(float)
    y_batch = y_train[:BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_cifar10_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_batch = x_train[:BATCH_SIZE].reshape((BATCH_SIZE, 3, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)).astype(float)
    y_batch = y_train[:BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


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


@pytest.fixture(scope="session", autouse=True)
def load_cnn_2d_mnist():
    """
    Load 2D CNN pre-trained on MNIST
    """
    model = CNN_2D_TF(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 10, num_channels=1)
    model.load_weights("tests/weights/cnn_2d_mnist_weights.keras")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_cnn_2d_cifar():
    """
    Load 2D CNN pre-trained on Cifar10
    """
    model = CNN_2D_TF(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, 10, num_channels=3)
    model.load_weights("tests/weights/cnn_2d_cifar_weights.keras")
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_tf():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_batch = x_train[:BATCH_SIZE, ...].reshape((BATCH_SIZE, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)).astype(float)
    y_batch = y_train[:BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_tf_mini():
    """
    Load mini batch of MNIST digits: inputs and outputs to use for testing
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_batch = x_train[:MINI_BATCH_SIZE, ...].reshape((MINI_BATCH_SIZE, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1)).astype(float)
    y_batch = y_train[:MINI_BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_mini():
    """
    Load mini batch of MNIST digits: inputs and outputs to use for testing
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_batch = x_train[:MINI_BATCH_SIZE, ...].reshape((MINI_BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)).astype(float)
    y_batch = y_train[:MINI_BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_cifar10_images_tf_mini():
    """
    Load mini batch of Cifar10 digits: inputs and outputs to use for testing
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_batch = x_train[:MINI_BATCH_SIZE].reshape((MINI_BATCH_SIZE, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, 3)).astype(float)
    y_batch = y_train[:MINI_BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session", autouse=True)
def load_cifar10_images_mini():
    """
    Load mini batch of Cifar10 digits: inputs and outputs to use for testing
    """
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_batch = x_train[:MINI_BATCH_SIZE, ...].reshape((MINI_BATCH_SIZE, 3, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)).astype(float)
    y_batch = y_train[:MINI_BATCH_SIZE].reshape(-1).astype(int)
    return {"x_batch": x_batch, "y_batch": y_batch}