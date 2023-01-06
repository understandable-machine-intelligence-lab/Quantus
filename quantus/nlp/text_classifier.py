from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


class TextClassifier(ABC):
    tokenizer: Tokenizer

    @abstractmethod
    def word_embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def forward_pass(
        self, inputs_embeds: tf.Tensor, attention_mask: Optional[tf.Tensor]
    ) -> tf.Tensor:
        # Must be able to record gradient
        pass

    @abstractmethod
    def predict(self, text: List[str]) -> np.ndarray:
        # Must be able to handle huge batches
        pass


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: List[str]) -> Dict[str, tf.Tensor] | tf.Tensor:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int] | tf.Tensor) -> List[str]:
        pass

    @abstractmethod
    def split_into_tokens(self, text: str) -> List[str]:
        pass


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, handle: str):
        self._tokenizer = AutoTokenizer.from_pretrained(handle)

    def tokenize(self, text: List[str]) -> Dict[str, tf.Tensor]:
        return self._tokenizer(text, padding="longest", return_tensors="tf").data

    def convert_ids_to_tokens(self, ids: tf.Tensor) -> List[str]:
        return self._tokenizer.convert_ids_to_tokens(ids.numpy())

    def split_into_tokens(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)


class HuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self,
        handle: str,
        embeddings_getter: Optional[
            Callable[[TFAutoModelForSequenceClassification], tf.Tensor]
        ] = None,
    ):
        self._embeddings_getter = embeddings_getter
        self._model = TFAutoModelForSequenceClassification.from_pretrained(handle)
        self.tokenizer = HuggingFaceTokenizer(handle)

    def word_embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        if self._embeddings_getter is None:
            raise ValueError("Please provider embeddings_getter argument to __init__")
        word_embeddings = self._embeddings_getter(self._model)
        return tf.nn.embedding_lookup(word_embeddings, input_ids)

    def forward_pass(
        self, inputs_embeds: tf.Tensor, attention_mask: Optional[tf.Tensor]
    ) -> tf.Tensor:
        return self._model(
            None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            training=False,
        ).logits

    def predict(self, text: List[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        return self._model.predict(tokens, verbose=0).logits


class KerasTokenizer(Tokenizer):
    def tokenize(self, text: List[str]) -> tf.Tensor:
        pass

    def convert_ids_to_tokens(self, ids: List[int] | tf.Tensor) -> List[str]:
        pass

    def split_into_tokens(self, text: str) -> List[str]:
        pass


class KerasTextClassifier(TextClassifier):
    def __init__(
        self,
        model: tf.keras.Model,
        tokenize_function: Callable[[List[str]], tf.Tensor],
        embeddings_getter: Optional[Callable[[tf.keras.Model], tf.Tensor]] = None,
    ):
        self._model = model
        self._embeddings_getter = embeddings_getter
        self._tokenize_function = tokenize_function

    def word_embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        return self._embeddings_getter(self._model)

    def forward_pass(
        self, inputs_embeds: tf.Tensor, attention_mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # TODO check args
        return self._model(None, input_embeddings=inputs_embeds, training=False)

    def predict(self, text: List[str]) -> np.ndarray:
        tokens = self._tokenize_function(text)
        return self._model.predict(tokens)
