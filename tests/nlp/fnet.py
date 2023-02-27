from __future__ import annotations

import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from keras_nlp.layers import FNetEncoder, TokenAndPositionEmbedding
from keras_nlp.tokenizers import WordPieceTokenizer
from typing import TYPE_CHECKING, List
from keras_nlp.tokenizers.word_piece_tokenizer import pretokenize
import numpy as np

if TYPE_CHECKING:
    from quantus.nlp import TF_TensorLike

import quantus.nlp as qn


MAX_SEQUENCE_LENGTH = 40
VOCAB_SIZE = 15000
EMBED_DIM = 128
INTERMEDIATE_DIM = 512
NUM_CLASSES = 4


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


class TokenizerAdapter(qn.Tokenizer):
    def __init__(self, tokenizer: WordPieceTokenizer):
        self._tokenizer = tokenizer

    def tokenize(self, text: List[str]) -> np.ndarray:
        return self._tokenizer(text).numpy()

    def convert_ids_to_tokens(self, ids: List[int] | np.ndarray) -> List[str]:
        return [self._tokenizer.id_to_token(i) for i in ids]

    def token_id(self, token: str) -> int:
        return self._tokenizer.token_to_id(token)


class FNetAdapter(qn.TextClassifier):
    def __init__(self, model: FNet, tokenizer: TokenizerAdapter):
        self._model = model
        self._tokenizer = tokenizer

    def __call__(self, inputs_embeds: TF_TensorLike, **kwargs) -> tf.Tensor:
        return self._model(
            tf.zeros(shape=tf.convert_to_tensor(tf.shape(inputs_embeds))[0]),
            training=False,
            input_embeddings=inputs_embeds,
        )

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        tokens = self._tokenizer.tokenize(text)
        return self._model.predict(tokens, verbose=0, batch_size=batch_size)

    def embedding_lookup(self, input_ids: TF_TensorLike, **kwargs) -> np.ndarray:
        token_embeds = self._model.embedding.token_embedding(input_ids)
        position_embeds = self._model.embedding.position_embedding(token_embeds)
        return token_embeds + position_embeds

    @property
    def weights(self):
        return self._model.get_weights()

    @weights.setter
    def weights(self, weights):
        self._model.set_weights(weights)

    @property
    def tokenizer(self) -> qn.Tokenizer:
        return self._tokenizer


    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator[Tuple[str, TextClassifier], None, None]:
        pass

    def get_hidden_representations(
        self,
        x_batch: List[str],
    ) -> np.ndarray:
        pass

    def get_hidden_representations_embeddings(
        self, x_batch: np.ndarray, **kwargs
    ) -> np.ndarray:
        pass


def fnet_adapter() -> FNetAdapter:
    with open("tests/assets/nlp/vocab_fnet_ag_news.txt", "r") as file:
        vocab = file.read().split("\n")

    tokenizer = WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=MAX_SEQUENCE_LENGTH,
    )

    model = FNet(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        num_classes=NUM_CLASSES,
    )

    model(tokenizer(["hello there"]))
    model.load_weights("tests/assets/nlp/weights_fnet_ag_news.keras")

    return FNetAdapter(model, TokenizerAdapter(tokenizer))
