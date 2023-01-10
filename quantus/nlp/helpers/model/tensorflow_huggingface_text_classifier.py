from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import List, Dict, Callable, Optional
from transformers import (
    TFPreTrainedModel,
    PreTrainedTokenizerBase,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    TFDistilBertForSequenceClassification,
    TFBertForSequenceClassification,
)
from transformers.tf_utils import shape_list
from .text_classifier import TextClassifier, Tokenizer


class HuggingFaceTokenizer(Tokenizer):

    tokenizer: PreTrainedTokenizerBase

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def tokenize(self, text: List[str]) -> Dict[str, np.ndarray]:
        return self.tokenizer(text, padding="longest", return_tensors="np").data

    def convert_ids_to_tokens(self, ids: np.ndarray) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def split_into_tokens(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


def _hf_bert_embeddings_lookup(
    model: TFBertForSequenceClassification, input_ids: tf.Tensor | np.ndarray
) -> tf.Tensor:
    word_embeds = tf.nn.embedding_lookup(
        tf.convert_to_tensor(model.bert.embeddings.weight, dtype=tf.float32), input_ids
    )
    input_shape = shape_list(word_embeds)[:-1]
    position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
    positional_embeds = tf.nn.embedding_lookup(
        tf.convert_to_tensor(
            model.bert.embeddings.position_embeddings, dtype=tf.float32
        ),
        position_ids,
    )
    return word_embeds + positional_embeds


def _hf_distilbert_embeddings_lookup(
    model: TFDistilBertForSequenceClassification, input_ids: tf.Tensor | np.ndarray
) -> tf.Tensor:
    word_embeds = tf.nn.embedding_lookup(
        tf.convert_to_tensor(model.distilbert.embeddings.weight, dtype=tf.float32),
        input_ids,
    )
    input_shape = shape_list(word_embeds)[:-1]
    position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
    positional_embeds = tf.nn.embedding_lookup(
        tf.convert_to_tensor(
            model.distilbert.embeddings.position_embeddings, dtype=tf.float32
        ),
        position_ids,
    )
    return word_embeds + positional_embeds


class HuggingFaceTextClassifierTF(TextClassifier):

    model: TFPreTrainedModel
    tokenizer: HuggingFaceTokenizer

    @staticmethod
    def from_pretrained(handle: str) -> HuggingFaceTextClassifierTF:
        if "distilbert" in handle:
            lookup_fn = _hf_distilbert_embeddings_lookup
        elif "bert" in handle:
            lookup_fn = _hf_bert_embeddings_lookup
        else:
            raise ValueError(f"Please provide embeddings_lookup_fn")

        return HuggingFaceTextClassifierTF(
            model=TFAutoModelForSequenceClassification.from_pretrained(handle),
            tokenizer=AutoTokenizer.from_pretrained(handle),
            embedding_lookup_fn=lookup_fn,
        )

    def __init__(
        self,
        model: TFPreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        embedding_lookup_fn: Callable[
            [TFPreTrainedModel, tf.Tensor | np.ndarray], tf.Tensor
        ],
    ):
        self._embedding_lookup_fn = embedding_lookup_fn
        self.model = model
        self.tokenizer = HuggingFaceTokenizer(tokenizer)

    def embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        if self._embedding_lookup_fn is None:
            raise ValueError("Please provider embedding_lookup_fn argument to __init__")
        return self._embedding_lookup_fn(self.model, input_ids)

    def forward(
        self, inputs_embeds: tf.Tensor, attention_mask: Optional[tf.Tensor]
    ) -> tf.Tensor:
        return self.model(
            None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            training=False,
        ).logits

    def predict(self, text: List[str]) -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        return self.model.predict(tokens, verbose=0).logits

    def set_weights(self, weights: List[np.ndarray]):
        self.model.set_weights(weights)

    def get_weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def get_attention_scores(self, text: List[str]) -> List[tf.Tensor]:
        tokens = self.tokenizer.tokenize(text)
        return self.model(tokens, output_attentions=True).attentions
