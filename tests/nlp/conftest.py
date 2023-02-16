import numpy as np
import pytest
from datasets import load_dataset

from quantus.nlp import TFHuggingFaceTextClassifier, TorchHuggingFaceTextClassifier

BATCH_SIZE = 8


@pytest.fixture(scope="session")
def sst2_dataset_huge_batch():
    ds = load_dataset("sst2")["test"]["sentence"]
    return ds[:1024]


@pytest.fixture(scope="session")
def sst2_dataset(sst2_dataset_huge_batch):
    return sst2_dataset_huge_batch[:BATCH_SIZE]


@pytest.fixture(scope="session")
def ag_news_dataset():
    ds = load_dataset("ag_news")["test"]["text"]
    return ds[:BATCH_SIZE]


@pytest.fixture(scope="session")
def tf_distilbert_sst2_model():
    return TFHuggingFaceTextClassifier.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


@pytest.fixture(scope="session")
def torch_distilbert_sst2_model():
    return TorchHuggingFaceTextClassifier.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
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
def fnet_ag_news_model():
    from .tf_fnet_text_classifier import fnet_text_classifier

    return fnet_text_classifier()
