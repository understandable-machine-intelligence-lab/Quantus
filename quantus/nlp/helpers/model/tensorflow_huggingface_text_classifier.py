from __future__ import annotations

import numpy as np
from typing import List, Dict
from functools import singledispatchmethod

from transformers import (
    TFPreTrainedModel,
    PreTrainedTokenizerBase,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tf_utils import shape_list
import tensorflow as tf
from quantus.nlp.config import USE_XLA
from quantus.nlp.helpers.model.tensorflow_text_classifier import (
    TensorFlowTextClassifier,
)

from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer


class TensorFlowHuggingFaceTextClassifier(
    TensorFlowTextClassifier, HuggingFaceTokenizer
):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, model: TFPreTrainedModel):
        super().__init__(tokenizer)
        self._model = model
        self._model._jit_compile = USE_XLA  # noqa
        self._word_embedding_matrix = tf.convert_to_tensor(
            self._model.get_input_embeddings().weight
        )
        self._position_embedding_matrix = tf.convert_to_tensor(
            self._model.get_input_embeddings().position_embeddings
        )

    @staticmethod
    def from_pretrained(handle: str, **kwargs) -> TensorFlowHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate TensorFlow models from HuggingFace Hub."""
        return TensorFlowHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            TFAutoModelForSequenceClassification.from_pretrained(handle, **kwargs),
        )

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

    def tokenize(self, text: List[str], **kwargs) -> Dict[str, np.ndarray]:
        return super().tokenize(text, return_tensors="tf", **kwargs)

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self.tokenize(text)
        return self._model.predict(encoded_inputs, verbose=0, **kwargs).logits

    @singledispatchmethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:  # type: ignore
        pass

    @get_hidden_representations.register
    def _(self, x_batch: list, **kwargs) -> tf.Tensor:
        encoded_batch = self.tokenize(x_batch)
        hidden_states = self._model(
            **encoded_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = tf.transpose(tf.stack(hidden_states), [1, 0, 2, 3])
        return hidden_states.numpy()

    @get_hidden_representations.register
    def _(self, x_batch: tf.Tensor, **kwargs) -> tf.Tensor:
        hidden_states = self._model(
            None,
            inputs_embeds=x_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = tf.transpose(tf.stack(hidden_states), [1, 0, 2, 3]).numpy()
        return hidden_states

    @property
    def internal_model(self) -> tf.keras.Model:
        return self._model

    def clone(self) -> TensorFlowTextClassifier:
        return self.from_pretrained(self._model.name_or_path)

    def embedding_lookup(self, input_ids: tf.Tensor, **kwargs) -> tf.Tensor:
        word_embeds = tf.nn.embedding_lookup(self._word_embedding_matrix, input_ids)
        input_shape = shape_list(word_embeds)[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        positional_embeds = tf.nn.embedding_lookup(
            self._position_embedding_matrix, position_ids
        )
        return word_embeds + positional_embeds
