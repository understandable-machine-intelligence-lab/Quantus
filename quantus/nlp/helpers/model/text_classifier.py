from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class TextClassifier(ABC):
    tokenizer: Tokenizer

    @abstractmethod
    def embedding_lookup(self, input_ids) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, inputs_embeds, attention_mask: Optional):
        # Must be able to record gradient
        pass

    @abstractmethod
    def predict(self, text: List[str]) -> np.ndarray:
        # Must be able to handle huge batches
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: List[str]) -> Dict[str, np.ndarray] | np.ndarray:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int] | np.ndarray) -> List[str]:
        pass

    @abstractmethod
    def split_into_tokens(self, text: str) -> List[str]:
        pass
