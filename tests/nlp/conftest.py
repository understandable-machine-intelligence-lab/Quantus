import pytest
from transformers import TFDistilBertForSequenceClassification
from datasets import load_dataset
from keras_nlp.tokenizers import WordPieceTokenizer
from ml_collections import ConfigDict
import tensorflow as tf

from .models import FNetClassifier
from quantus.nlp.text_classifier import HuggingFaceTextClassifier

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
    def getter(hm: TFDistilBertForSequenceClassification):
        return tf.convert_to_tensor(hm.distilbert.embeddings.weight, dtype=tf.float32)

    model = HuggingFaceTextClassifier(
        "distilbert-base-uncased-finetuned-sst-2-english", embeddings_getter=getter
    )
    return model


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
def fnet_ag_news_model(fnet_config):
    with open(fnet_config.vocab_path, "r") as file:
        vocab = file.read().split("\n")

    tokenizer = WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=fnet_config.max_sequence_length,
    )
    model = FNetClassifier(fnet_config, 4)
