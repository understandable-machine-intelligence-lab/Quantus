from __future__ import annotations

import numpy as np
from typing import List
from multimethod import multimethod

from transformers import (
    TFPreTrainedModel,
    PreTrainedTokenizerBase,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tf_utils import shape_list
import tensorflow as tf
from quantus.nlp.helpers.model.tensorflow_text_classifier import (
    TensorFlowTextClassifier,
)
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer


class TensorFlowHuggingFaceTextClassifier(TensorFlowTextClassifier):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: TFPreTrainedModel):
        self._model = model
        self._tokenizer = HuggingFaceTokenizer(tokenizer)
        self._word_embedding_matrix = tf.convert_to_tensor(
            self._model.get_input_embeddings().weight
        )
        self._position_embedding_matrix = tf.convert_to_tensor(
            self._model.get_input_embeddings().position_embeddings
        )

    @staticmethod
    def from_pretrained(handle: str) -> TensorFlowHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate TensorFlow models from HuggingFace Hub."""
        return TensorFlowHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            TFAutoModelForSequenceClassification.from_pretrained(handle),
        )

    def embedding_lookup(self, input_ids: np.ndarray, **kwargs) -> tf.Tensor:
        word_embeds = tf.nn.embedding_lookup(self._word_embedding_matrix, input_ids)
        input_shape = shape_list(word_embeds)[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        positional_embeds = tf.nn.embedding_lookup(
            self._position_embedding_matrix, position_ids
        )
        return word_embeds + positional_embeds

    def __call__(
        self,
        inputs_embeds: tf.Tensor,
        **kwargs,
    ) -> tf.Tensor:
        return self._model(
            None,
            inputs_embeds=inputs_embeds,
            training=False,
            **kwargs,
        ).logits

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(text)
        return self._model.predict(encoded_inputs, verbose=0, **kwargs).logits

    @multimethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:
        pass

    @get_hidden_representations.register
    def _(self, x_batch: List[str], **kwargs) -> np.ndarray:
        encoded_batch = self._tokenizer.tokenize(x_batch)
        hidden_states = self._model(
            **encoded_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states

    @get_hidden_representations.register
    def _(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        hidden_states = self._model(
            None,
            inputs_embeds=x_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def internal_model(self) -> tf.keras.Model:
        return self._model

    def clone(self) -> TensorFlowTextClassifier:
        return self.from_pretrained(self._model.name_or_path)
