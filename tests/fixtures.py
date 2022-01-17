from typing import Tuple
import os
import pytest
import pickle
import torch
import torchvision
from torchvision import transforms
import numpy as np
from ..quantus.helpers.models import LeNet, LeNetTF
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.image import grayscale_to_rgb


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model_tf():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetTF()
    model.load_weights('tutorials/assets/mnist_tf_weights/')
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


@pytest.fixture(scope="session", autouse=True)
def load_mnist_images_tf():
    # TODO: save data in the assets folder and load just one batch
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(124)
    for images, labels in ds_test.take(1):
        x_batch = images.numpy()
        y_batch = labels.numpy()
    return {"x_batch": to_rgb(x_batch), "y_batch": y_batch}


@pytest.fixture
def almost_uniform(scope="session", autouse=True):
    a_batch = np.random.uniform(0, 0.01, size=(10, 1, 224, 224))
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": a_batch,
    }


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def to_rgb(x):
    return grayscale_to_rgb(tf.expand_dims(x, axis=3))
