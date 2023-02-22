from __future__ import annotations

import numpy as np
from typing import List, Optional, TypeVar, Generator, Any, Tuple
from abc import ABC, abstractmethod
from quantus.nlp.helpers.model.tokenizer import Tokenizer

T = TypeVar("T")


class TextClassifier(ABC):
    tokenizer: Tokenizer

    """An interface for model, trained for text-classification task (aka sentiment analysis)."""

    @abstractmethod
    def __call__(self, inputs_embeds: T, attention_mask: Optional[T], **kwargs) -> T:
        """
        Execute forward pass on latent representation for input tokens.
        This method must return tensors of corresponding DNN framework.
        This method must be able to record gradients, as it potentially will be used by gradient based XAI methods.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        """
        Execute forward pass with plain text inputs, return logits as np.ndarray.
        This method must be able to handle huge batches, as it potentially could be called with entire dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_lookup(self, input_ids, **kwargs):
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self):
        """Get model's (learnable) parameters."""
        raise NotImplementedError

    @weights.setter
    @abstractmethod
    def weights(self, weights):
        """Set model's (learnable) parameters."""
        raise NotImplementedError

    @abstractmethod
    def get_hidden_representations(
        self,
        x_batch: List[str],
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_hidden_representations_embeddings(
        self,
        x_batch: np.ndarray,
        attention_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42, **kwargs
    ) -> Generator[Tuple[Any, TextClassifier], None, None]:
        raise NotImplementedError
