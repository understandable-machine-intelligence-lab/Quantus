import numpy as np
import pytest
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
from keras_nlp.tokenizers import WordPieceTokenizer
from ml_collections import ConfigDict
import tensorflow as tf

from .models import FNetClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    HuggingFaceTextClassifierTF,
    HuggingFaceTokenizer,
)

BATCH_SIZE = 8


@pytest.fixture(scope="session")
def sst2_dataset():
    ds = load_dataset("sst2")["test"]["sentence"]
    return ds[:BATCH_SIZE]


@pytest.fixture(scope="session")
def ag_news_dataset():
    ds = load_dataset("ag_news")["test"]["text"]
    return ds[:BATCH_SIZE]


@pytest.fixture(scope="session")
def distilbert_sst2_model():
    return HuggingFaceTextClassifierTF.from_pretrained(
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
def a_tuple_text_ragged_1(a_tuple_text):
    return (
        (list(a_tuple_text[0][0])[:37], a_tuple_text[0][1][:37]),
        (list(a_tuple_text[1][0]), a_tuple_text[1][1]),
    )


@pytest.fixture(scope="session")
def a_tuple_text_ragged_2(a_tuple_text):
    return (
        (list(a_tuple_text[0][0]), a_tuple_text[0][1]),
        (list(a_tuple_text[1][0])[:37], a_tuple_text[1][1][:37]),
    )


@pytest.fixture(scope="session")
def fnet_config() -> ConfigDict:
    config = ConfigDict()
    config.max_sequence_length = 512
    config.vocab_size = 15000
    config.embed_dim = 128
    config.intermediate_dim = 512
    config.vocab_path = "assets/nlp/vocab_fnet_ag_news.txt"
    return config


@pytest.fixture(scope="session")
def fnet_ag_news_model_tokenizer(fnet_config):
    with open(fnet_config.vocab_path, "r") as file:
        vocab = file.read().split("\n")

    return WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=fnet_config.max_sequence_length,
    )


@pytest.fixture(scope="session")
def fnet_ag_news_model(fnet_config):
    model = FNetClassifier(fnet_config, 4)
    return model
