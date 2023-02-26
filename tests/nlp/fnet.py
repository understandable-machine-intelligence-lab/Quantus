from __future__ import annotations

import tensorflow as tf
from ml_collections import ConfigDict
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from keras_nlp.layers import FNetEncoder, TokenAndPositionEmbedding
from keras_nlp.tokenizers import WordPieceTokenizer
from typing import TYPE_CHECKING, List
from keras_nlp.tokenizers.word_piece_tokenizer import pretokenize
import numpy as np

if TYPE_CHECKING:
    from quantus.nlp import TF_TensorLike  # noqa

import quantus.nlp as qn  # noqa


class FNet(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        embedding_dim: int,
        intermediate_dim: int,
        num_classes: int,
    ):
        super(FNet, self).__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            mask_zero=True,
        )
        self.encoder1 = FNetEncoder(intermediate_dim)
        self.encoder2 = FNetEncoder(intermediate_dim)
        self.encoder3 = FNetEncoder(intermediate_dim)

        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(0.1)
        self.top = Dense(num_classes)

    def call(self, inputs, training=None, input_embeddings=None):
        if input_embeddings is None:
            x = self.embedding(inputs)
        else:
            x = input_embeddings

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        return self.top(x)


class FNetTokenizerAdapter(qn.Tokenizer):
    tokenizer: WordPieceTokenizer

    def __init__(self, tokenizer: WordPieceTokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text: List[str]) -> np.ndarray:
        return self.tokenizer(text).numpy()

    def convert_ids_to_tokens(self, ids: List[int] | np.ndarray) -> List[str]:
        return [self.tokenizer.id_to_token(i) for i in ids]

    def split_into_tokens(self, text: str) -> List[str]:
        tokens = (
            pretokenize(
                tf.convert_to_tensor(text),
                split=self.tokenizer.split,
                split_on_cjk=self.tokenizer.split_on_cjk,
                strip_accents=self.tokenizer.strip_accents,
                lowercase=self.tokenizer.lowercase,
            )
            .numpy()
            .tolist()
        )

        tokens = tokens[0]
        return [i.decode("utf-8") for i in tokens]


class FNetTextClassifierAdapter(qn.TextClassifier):
    tokenizer: FNetTokenizerAdapter
    model: FNet

    def __init__(self, model: FNet, tokenizer: FNetTokenizerAdapter):
        self.model = model
        self._tokenizer = tokenizer

    def embedding_lookup(self, input_ids: TF_TensorLike, **kwargs) -> np.ndarray:
        token_embeds = self.model.embedding.token_embedding(input_ids)
        position_embeds = self.model.embedding.position_embedding(token_embeds)
        return token_embeds + position_embeds

    def __call__(self, inputs_embeds: TF_TensorLike, *args, **kwargs) -> tf.Tensor:
        return self.model(
            tf.zeros(shape=tf.convert_to_tensor(tf.shape(inputs_embeds))[0]),
            training=False,
            input_embeddings=inputs_embeds,
        )

    def predict(self, text: List[str], batch_size: int = 64) -> np.ndarray:
        tokens = self._tokenizer.tokenize(text)
        return self.model.predict(tokens, verbose=0, batch_size=batch_size)

    @property
    def weights(self):
        return self.model.get_weights()

    @weights.setter
    def weights(self, weights):
        self.model.set_weights(weights)

    @property
    def tokenizer(self) -> qn.Tokenizer:
        return self._tokenizer


def fnet_adapter(config: ConfigDict, dataset: str) -> FNetTextClassifierAdapter:
    if dataset == "sst2":
        vocab_path = config.sst2_vocab_path
        num_classes = 2
        weights_path = config.sst2_weights_path

    elif dataset == "ag_news":
        vocab_path = config.ag_news_vocab_path
        num_classes = 4
        weights_path = config.ag_news_vocab_path
    else:
        raise ValueError()

    with open(vocab_path, "r") as file:
        vocab = file.read().split("\n")

    tokenizer = WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=config.max_sequence_length,
    )

    model = FNet(
        vocabulary_size=config.vocab_size,
        sequence_length=config.max_sequence_length,
        embedding_dim=config.embed_dim,
        intermediate_dim=config.intermediate_dim,
        num_classes=num_classes,
    )

    model(tokenizer(["hello there"]))
    model.load_weights(weights_path)

    return FNetTextClassifierAdapter(
        model=model, tokenizer=FNetTokenizerAdapter(tokenizer)
    )
