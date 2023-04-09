import pickle
from importlib import util

import numpy as np
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

# Set seed for reproducibility.
np.random.seed(42)


@pytest.fixture(scope="session")
def load_mnist_images_tf(load_mnist_images):
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    return {
        "x_batch": np.moveaxis(load_mnist_images["x_batch"], 1, -1),
        "y_batch": load_mnist_images["y_batch"],
    }


@pytest.fixture(scope="session")
def almost_uniform_1d():
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 100)),
    }


@pytest.fixture(scope="session")
def almost_uniform_1d_no_abatch_channel_last():
    return {
        "x_batch": np.random.randn(10, 100, 3),
        "y_batch": np.random.randint(0, 10, size=10),
    }


@pytest.fixture(scope="session")
def almost_uniform_1d_no_abatch():
    return {
        "x_batch": np.random.randn(10, 3, 100),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture(scope="session")
def almost_uniform_2d():
    return {
        "x_batch": np.random.randn(10, 3, 224, 224),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": np.random.uniform(0, 0.01, size=(10, 1, 224, 224)),
    }


@pytest.fixture(scope="session")
def almost_uniform_2d_no_abatch():
    return {
        "x_batch": np.random.randn(10, 1, 28, 28),
        "y_batch": np.random.randint(0, 10, size=10),
        "a_batch": None,
    }


@pytest.fixture(scope="session")
def flat_image_array():
    return {
        "x": np.zeros((1, 3 * 28 * 28)),
        "shape": (3, 28, 28),
        "channel_first": True,
    }


@pytest.fixture(scope="session")
def flat_sequence_array():
    return {
        "x": np.zeros((1, 3 * 28)),
        "shape": (3, 28),
        "channel_first": True,
    }


def sst2_dataset():
    x_batch = np.load("tests/assets/cifar10/x_batch.npy")
    y_batch = np.load("tests/assets/cifar10/y_batch.npy")
    return {"x_batch": x_batch, "y_batch": y_batch}


if util.find_spec("tensorflow"):
    from quantus.helpers.model.models import (
        LeNetTF,
        CifarCNNModel,
        ConvNet1DTF,
        TitanicSimpleTFModel,
    )

    @pytest.fixture(scope="session")
    def load_mnist_model_tf():
        """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
        model = LeNetTF()
        model.load_weights("tests/assets/weights/lenet_mnist.keras")
        return model

    @pytest.fixture(scope="session")
    def load_cifar10_model_tf():
        """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
        model = CifarCNNModel()
        model.load_weights("tests/assets/weights/cifar_cnn.keras")
        return model

    @pytest.fixture(scope="session")
    def load_1d_3ch_conv_model_tf():
        """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
        model = ConvNet1DTF(n_channels=3, seq_len=100, n_classes=10)
        # TODO: add trained model weights
        # model = LeNetTF()
        # model.load_weights("tests/assets/mnist_tf_weights/")
        return model

    @pytest.fixture(scope="session")
    def titanic_model_tf(titanic_dataset):
        model = TitanicSimpleTFModel()
        model(titanic_dataset["x_batch"], training=False)
        model.load_weights("tests/assets/weights/titanic.keras")
        return model


if util.find_spec("torch"):
    import torch
    from quantus.helpers.model.models import (
        LeNet,
        ConvNet1D,
        ConvNet1DTF,
        TitanicSimpleTorchModel,
    )

    @pytest.fixture(scope="session")
    def load_mnist_model():
        """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
        model = LeNet()
        model.load_state_dict(
            torch.load(
                "tests/assets/weights/mnist.pickle",
                map_location="cpu",
                pickle_module=pickle,
            )
        )
        return model

    @pytest.fixture(scope="session")
    def load_1d_1ch_conv_model():
        """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
        model = ConvNet1D(n_channels=1, n_classes=10)
        model.eval()
        # TODO: add trained model weights
        # model.load_state_dict(
        #    torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
        # )
        return model

    @pytest.fixture(scope="session")
    def load_1d_3ch_conv_model():
        """Load a pre-trained 1d-convolutional classification model (architecture at quantus/helpers/models)."""
        model = ConvNet1D(n_channels=3, n_classes=10)
        model.eval()
        # TODO: add trained model weights
        # model.load_state_dict(
        #    torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
        # )
        return model

    @pytest.fixture(scope="session")
    def titanic_model_torch():
        model = TitanicSimpleTorchModel()
        model.load_state_dict(torch.load("tests/assets/weights/titanic.pickle"))
        return model


@pytest.fixture(scope="session")
def load_mnist_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = np.load("tests/assets/mnist/x_batch.npy")
    y_batch = np.load("tests/assets/mnist/y_batch.npy")
    return {"x_batch": x_batch, "y_batch": y_batch}


@pytest.fixture(scope="session")
def load_cifar10_images():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = np.load("tests/assets/cifar10/x_batch.npy")
    y_batch = np.load("tests/assets/cifar10/y_batch.npy")
    return {"x_batch": x_batch, "y_batch": y_batch}


# Just some aliases to avoid long names in parameterized tests.
# TODO
# ---------------- NLP fixtures ------------------


@pytest.fixture(scope="session")
def sst2_dataset():
    x_batch = np.load("tests/assets/sst2/x_batch.npy").tolist()
    y_batch = np.load("tests/assets/sst2/y_batch.npy")
    return {"x_batch": x_batch, "y_batch": y_batch}


if util.find_spec("transformers"):
    from quantus.helpers.utils import get_wrapped_text_classifier
    from transformers import (
        TFDistilBertForSequenceClassification,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )

    @pytest.fixture(scope="session")
    def tf_sst2_model():
        return TFDistilBertForSequenceClassification.from_pretrained(
            "tests/assets/distilbert/"
        )

    @pytest.fixture(scope="session")
    def torch_sst2_model():
        return DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    @pytest.fixture(scope="session")
    def sst2_tokenizer():
        return DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    @pytest.fixture(scope="session")
    def tf_sst2_model_wrapper(tf_sst2_model, sst2_tokenizer):
        return get_wrapped_text_classifier(tf_sst2_model, sst2_tokenizer)

    @pytest.fixture(scope="session")
    def torch_sst2_model_wrapper(torch_sst2_model, sst2_tokenizer):
        return get_wrapped_text_classifier(torch_sst2_model, sst2_tokenizer)


@pytest.fixture(scope="session")
def titanic_dataset():
    df = pd.read_csv("tutorials/assets/titanic3.csv")
    df = df[["age", "embarked", "fare", "parch", "pclass", "sex", "sibsp", "survived"]]
    df["age"] = df["age"].fillna(df["age"].mean())
    df["fare"] = df["fare"].fillna(df["fare"].mean())

    df_enc = pd.get_dummies(df, columns=["embarked", "pclass", "sex"]).sample(frac=1)
    X = df_enc.drop(["survived"], axis=1).values.astype(float)
    Y = df_enc["survived"].values.astype(int)
    _, test_features, _, test_labels = train_test_split(X, Y, test_size=0.3)
    return {"x_batch": test_features, "y_batch": test_labels}
