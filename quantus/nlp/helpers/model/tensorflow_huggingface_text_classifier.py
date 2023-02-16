from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

from transformers import (
    TFPreTrainedModel,
    PreTrainedTokenizerBase,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.tf_utils import shape_list
import tensorflow as tf
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TF_TensorLike


class TFHuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: TFPreTrainedModel,
    ):
        self.model = model
        self.tokenizer = HuggingFaceTokenizer(tokenizer)
        self.word_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().weight, dtype=tf.float32
        )
        self.position_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().position_embeddings, dtype=tf.float32
        )

    @staticmethod
    def from_pretrained(handle: str) -> TFHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate TensorFlow models from HuggingFace Hub."""
        return TFHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            TFAutoModelForSequenceClassification.from_pretrained(handle),
        )

    def embedding_lookup(self, input_ids: TF_TensorLike, **kwargs) -> tf.Tensor:
        word_embeds = tf.nn.embedding_lookup(self.word_embedding_matrix, input_ids)
        input_shape = shape_list(word_embeds)[:-1]
        position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        positional_embeds = tf.nn.embedding_lookup(
            self.position_embedding_matrix, position_ids
        )
        return word_embeds + positional_embeds

    def __call__(
        self,
        inputs_embeds: TF_TensorLike,
        attention_mask: Optional[TF_TensorLike],
        **kwargs,
    ) -> tf.Tensor:
        return self.model(
            None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            training=False,
        ).logits

    def predict(self, text: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        return self.model.predict(
            self.tokenizer.tokenize(text), verbose=0, batch_size=batch_size, **kwargs
        ).logits

    @property
    def weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    @weights.setter
    def weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)
