from __future__ import annotations

import tensorflow as tf
from typing import List, Dict, Iterable, Optional
from abc import ABC, abstractmethod


class TextClassifier(ABC):
    @abstractmethod
    def embedding_lookup(self, input_ids) -> tf.Tensor:
        pass

    @abstractmethod
    def forward_pass(
        self, inputs_embeds: tf.Tensor, attention_mask: Optional[tf.Tensor]
    ) -> tf.Tensor:
        pass


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: List[str]) -> Dict[str, tf.Tensor] | tf.Tensor:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        pass
