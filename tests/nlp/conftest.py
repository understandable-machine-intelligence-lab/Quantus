import numpy as np
import pytest
from datasets import load_dataset
import platform

if platform.system() != "Darwin" or platform.processor() != "arm":
    from tests.nlp.fnet import fnet_adapter
from quantus.nlp import TFHuggingFaceTextClassifier, TorchHuggingFaceTextClassifier

BATCH_SIZE = 8


@pytest.fixture(scope="session")
def sst2_dataset():
    dataset = load_dataset("sst2")["test"]["sentence"]
    return dataset[:BATCH_SIZE]


@pytest.fixture(scope="session")
def tf_sst2_model():
    return TFHuggingFaceTextClassifier.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


@pytest.fixture(scope="session")
def emotion_dataset():
    dataset = load_dataset("SetFit/emotion")["test"]["text"]
    return dataset[:BATCH_SIZE]


@pytest.fixture(scope="session")
def emotion_model():
    return TorchHuggingFaceTextClassifier.from_pretrained(
        "michellejieli/emotion_text_classifier"
    )


@pytest.fixture(scope="session")
def a_tuple_text():
    a_tuple = np.load("tests/assets/nlp/a_tuple_text.npy")
    return (
        (list(a_tuple[0][0]), a_tuple[0][1].astype(float)),
        (list(a_tuple[1][0]), a_tuple[1][1].astype(float)),
    )


@pytest.fixture(scope="session")
def a_batch_text():
    a_batch = np.load("tests/assets/nlp/a_batch_text.npy")

    def map_fn(a):
        return a[0], a[1].astype(float)

    return list(map(map_fn, a_batch))


@pytest.fixture(scope="session")
def a_tuple_text_ragged_1(a_tuple_text):
    return (
        (list(a_tuple_text[0][0])[:37], a_tuple_text[0][1][:37]),
        (list(a_tuple_text[1][0]), a_tuple_text[1][1]),
    )


@pytest.fixture(scope="session")
def a_tuple_text_ragged_2(a_tuple_text):
    return (
        (list(a_tuple_text[0][0]), a_tuple_text[0][1]),
        (list(a_tuple_text[1][0][:37]), a_tuple_text[1][1][:37]),
    )


@pytest.fixture(scope="session")
def torch_fnet():
    # This model is interesting because it has not attention mask, but requires type_ids
    return TorchHuggingFaceTextClassifier.from_pretrained(
        "gchhablani/fnet-base-finetuned-sst2"
    )


@pytest.fixture(scope="session")
def fnet_keras():
    # This model is interesting because the tokenizer doesn't return dict
    return fnet_adapter()


@pytest.fixture(scope="session")
def ag_news_dataset():
    return load_dataset("ag_news")["test"]["text"][:BATCH_SIZE]
