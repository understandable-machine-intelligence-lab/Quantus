import pytest
from datasets import load_dataset

from quantus.nlp import (
    TFHuggingFaceTextClassifier,
    TorchHuggingFaceTextClassifier,
)

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
def torch_sst2_model():
    return TorchHuggingFaceTextClassifier.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
