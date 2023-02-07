from __future__ import annotations
from typing import List, TYPE_CHECKING

import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, Dense, Dropout
from keras_nlp.layers import FNetEncoder, TokenAndPositionEmbedding
from keras_nlp.tokenizers import WordPieceTokenizer
from keras_nlp.tokenizers.word_piece_tokenizer import pretokenize
from typing import Optional


from quantus.nlp import TextClassifier, Tokenizer

if TYPE_CHECKING:
    from quantus.nlp import TF_TensorLike


MaxSequenceLength = 512
VocabSize = 15000
EmbeddingDim = 128
IntermediateDim = 512
VocabPath = "tests/assets/nlp/vocab_fnet_ag_news.txt"
WeightsPath = "tests/assets/nlp/weights_fnet_ag_news.keras"
NumClasses = 4


def fnet_text_classifier() -> TextClassifier:
    with open(VocabPath, "r") as file:
        vocab = file.read().split("\n")

    tokenizer = WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=MaxSequenceLength,
    )

    model = FNetClassifier(
        vocab_size=VocabSize,
        max_sequence_length=MaxSequenceLength,
        embed_dim=EmbeddingDim,
        intermediate_dim=IntermediateDim,
        num_classes=NumClasses,
    )

    model(tokenizer(["hello there"]))
    model.load_weights(WeightsPath)

    return FNetTextClassifierAdapter(
        model=model, tokenizer=FNetTokenizerAdapter(tokenizer)
    )


class FNetClassifier(tf.keras.Model):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_sequence_length: int,
        embed_dim: int,
        intermediate_dim: int,
        num_classes: int,
    ):
        super(FNetClassifier, self).__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=vocab_size,
            sequence_length=max_sequence_length,
            embedding_dim=embed_dim,
            mask_zero=True,
        )
        self.encoder1 = FNetEncoder(intermediate_dim)
        self.encoder2 = FNetEncoder(intermediate_dim)
        self.encoder3 = FNetEncoder(intermediate_dim)

        self.pool = GlobalAveragePooling1D()
        self.dropout = Dropout(0.1)
        self.top = Dense(num_classes)

    def call(
        self,
        inputs: TF_TensorLike,
        training: Optional[bool] = None,
        input_embeddings: Optional[TF_TensorLike] = None,
    ) -> tf.Tensor:
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


class FNetTokenizerAdapter(Tokenizer):
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


class FNetTextClassifierAdapter(TextClassifier):
    tokenizer: FNetTokenizerAdapter
    model: FNetClassifier

    def __init__(self, model: FNetClassifier, tokenizer: FNetTokenizerAdapter):
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
    def weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    @weights.setter
    def weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
