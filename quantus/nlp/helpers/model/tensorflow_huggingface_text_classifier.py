import numpy as np
import tensorflow as tf
from typing import List, Dict, Callable, Optional
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase
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


class HuggingFaceTextClassifierTF(TextClassifier):

    model: TFPreTrainedModel
    tokenizer: Tokenizer

    def __init__(
        self,
        model: TFPreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        embeddings_getter: Optional[Callable[[TFPreTrainedModel], tf.Tensor]] = None,
    ):
        self._embeddings_getter = embeddings_getter
        self.model = model
        self.tokenizer = HuggingFaceTokenizer(tokenizer)

    def word_embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        if self._embeddings_getter is None:
            raise ValueError("Please provider embeddings_getter argument to __init__")
        word_embeddings = self._embeddings_getter(self.model)
        return tf.nn.embedding_lookup(word_embeddings, input_ids)

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
