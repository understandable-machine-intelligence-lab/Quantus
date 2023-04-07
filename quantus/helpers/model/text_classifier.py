# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TypedDict, Tuple, Dict, Any, overload, TYPE_CHECKING

import numpy as np

from quantus.helpers.collection_utils import safe_as_array
from quantus.helpers.model.model_interface import RandomisableModel, HiddenRepresentationsModel, ModelWrapper

if TYPE_CHECKING:
    import tensorflow as tf
    import torch

R = TypedDict("R", {"input_ids": np.ndarray}, total=False)


class Tokenizable(ABC):


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
        """Convert vocabulary ids to strings."""
        raise NotImplementedError

    def get_input_ids(self, x_batch: List[str]) -> Tuple[Any, Dict[str, Any]]:
        """Do batch encode, unpack input ids and other forward-pass kwargs."""
        encoded_input = self.batch_encode(x_batch)
        return encoded_input.pop("input_ids"), encoded_input  # type: ignore


class TextClassifier(HiddenRepresentationsModel, RandomisableModel, ModelWrapper):
    """
    An interface for model, trained for text-classification task (aka sentiment analysis).
    TextClassifier is a model with signature F: List['text'] -> np.ndarray
    """

    tokenizer: Tokenizable

    def __call__(self, *args, **kwargs):
        return self.get_model()(*args, **kwargs)

    def get_embeddings(self, x_batch: List[str]) -> Tuple[np.ndarray, Dict[str, ...]]:
        """Do batch encode, unpack input ids, convert to embeddings."""
        input_ids, predict_kwargs = self.tokenizer.get_input_ids(x_batch)
        return safe_as_array(self.embedding_lookup(input_ids)), predict_kwargs

    @abstractmethod
    @overload
    def embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError

    @abstractmethod
    @overload
    def embedding_lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError

    @abstractmethod
    def embedding_lookup(self, input_ids: np.ndarray) -> np.ndarray:
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError

    @abstractmethod
    @overload
    def predict(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        """Execute forward pass on latent representation for input tokens."""
        raise NotImplementedError

    @abstractmethod
    @overload
    def predict(self, x_batch: tf.Tensor, **kwargs) -> tf.Tensor:
        """Execute forward pass on latent representation for input tokens."""
        raise NotImplementedError

    @abstractmethod
    @overload
    def predict(self, x_batch: torch.Tensor, **kwargs) -> torch.Tensor:
        """Execute forward pass on latent representation for input tokens."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch: List[str], batch_size=64, **kwargs) -> np.ndarray:
        """Execute forward pass with plain text inputs."""
        raise NotImplementedError

    @abstractmethod
    @overload
    def get_hidden_representations(
            self,
            x: List[str],
            *args,
            **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_hidden_representations(
            self,
            x: np.ndarray,
            *args,
            **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

# ---------- QA, NLI, Text Generation, Summarization, NER and more models to follow ----------
# Or actually no, Quantus is designed for classifiers, so I probably will just show examples of other tasks separately.
