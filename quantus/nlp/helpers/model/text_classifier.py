from __future__ import annotations

import numpy as np
from typing import List, TypeVar, Generator, Tuple
from abc import ABC, abstractmethod

T = TypeVar("T")


class TextClassifier(ABC):

    """An interface for model, trained for text-classification task (aka sentiment analysis)."""

    @property
    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def weights(self):
        """Get model's (learnable) parameters."""
        raise NotImplementedError  # pragma: not covered

    @weights.setter
    @abstractmethod
    def weights(self, weights):
        """Set model's (learnable) parameters."""
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def __call__(self, inputs_embeds: T, **kwargs) -> T:
        """
        Execute forward pass on latent representation for input tokens.
        This method must return tensors of corresponding DNN framework.
        This method must be able to record gradients, as it is used internally by gradient based XAI methods.
        """
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        """
        Execute forward pass with plain text inputs, return logits as np.ndarray.
        This method must be able to handle huge batches, as it potentially could be called with entire dataset.
        """
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def embedding_lookup(self, input_ids, **kwargs):
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def get_hidden_representations(
        self, x_batch: List[str] | np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Returns model's internal representations for x_batch.
        This method is required for Relative Representation Stability.
        """
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator[Tuple[str, TextClassifier], None, None]:
        """
        Yields layer name, and new model with this layer perturbed.
        This method is required for Model Parameter Randomisation Metric.
        """
        raise NotImplementedError  # pragma: not covered
