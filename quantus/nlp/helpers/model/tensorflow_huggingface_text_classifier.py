from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING, Generator

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
from quantus.nlp.helpers.utils import unpack_token_ids_and_attention_mask

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TF_TensorLike  # pragma: not covered


class TFHuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, model: TFPreTrainedModel, handle: str
    ):
        self.model = model
        self.tokenizer = HuggingFaceTokenizer(tokenizer)
        self.word_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().weight, dtype=tf.float32
        )
        self.position_embedding_matrix = tf.convert_to_tensor(
            model.get_input_embeddings().position_embeddings, dtype=tf.float32
        )
        self.handle = handle

    @staticmethod
    def from_pretrained(handle: str) -> TFHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate TensorFlow models from HuggingFace Hub."""
        return TFHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            TFAutoModelForSequenceClassification.from_pretrained(handle),
            handle,
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
            **kwargs,
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

    def get_random_layer_generator(
        self,
        order: str = "top_down",
        seed: int = 42,
    ) -> Generator:
        original_weights = self.model.get_weights().copy()
        random_layer_model = TFAutoModelForSequenceClassification.from_pretrained(
            self.handle
        )
        layers = list(self.model._flatten_layers(include_self=False, recursive=True))

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                random_layer_model.set_weights(original_weights)
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            random_layer_model_wrapper = TFHuggingFaceTextClassifier(
                AutoTokenizer.from_pretrained(self.handle),
                random_layer_model,
                self.handle,
            )
            yield layer.name, random_layer_model_wrapper

    def get_hidden_representations(self, x_batch: List[str]) -> np.ndarray:
        inputs_ids, attention_mask = unpack_token_ids_and_attention_mask(
            self.tokenizer.tokenize(x_batch)
        )
        x_batch_embeddings = self.embedding_lookup(inputs_ids)
        return self.get_hidden_representations_embeddings(
            x_batch_embeddings, attention_mask
        )

    def get_hidden_representations_embeddings(
        self, x_batch: np.ndarray, attention_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        hidden_states = self.model(
            None,
            attention_mask,
            inputs_embeds=x_batch,
            training=False,
            output_hidden_states=True,
        ).hidden_states
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states
