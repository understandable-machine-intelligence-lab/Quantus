from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator, List, Tuple, TypedDict, TypeVar

import numpy as np

T = TypeVar("T")
R = TypedDict("R", {"input_ids": np.ndarray}, total=False)


class TextClassifier(ABC):

    """An interface for model, trained for text-classification task (aka sentiment analysis)."""

    @abstractmethod
    def batch_encode(self, text: List[str], **kwargs) -> R:
        """Convert batch of plain-text inputs to vocabulary id's."""
        raise NotImplementedError

    @abstractmethod
    def convert_ids_to_tokens(self, ids: np.ndarray) -> List[str]:
        """Convert batch of vocabulary id's batch to batch of plain-text strings."""
        raise NotImplementedError

    @abstractmethod
    def token_id(self, token: str) -> int:
        """Get id of token. This method is required for TokenPruning metric."""
        raise NotImplementedError

    @abstractmethod
    def batch_decode(self, ids: np.ndarray, **kwargs) -> List[str]:
        pass

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
    def __call__(self, inputs_embeds: T, **kwargs) -> T:
        """
        Execute forward pass on latent representation for input tokens.
        This method must return tensors of corresponding DNN framework.
        This method must be able to record gradients, as it is used internally by gradient based XAI methods.
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

    @abstractmethod
    def get_hidden_representations(
        self, x_batch: List[str] | np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Returns model's internal representations for x_batch.
        This method is required for Relative Representation Stability.
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator[Tuple[str, TextClassifier], None, None]:
        """
        Yields layer name, and new model with this layer perturbed.
        This method is required for Model Parameter Randomisation Metric.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def random_layer_generator_length(self) -> int:
        raise NotImplementedError
