# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TypedDict, TypeVar, Tuple, Dict, Any

import numpy as np

from quantus.helpers.model.model_interface import (
    RandomisableModel,
    HiddenRepresentationsModel,
    ModelWrapper,
)

T = TypeVar("T")
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


class EmbeddingsCallable(ABC):

    @abstractmethod
    def __call__(self, inputs_embeds: T, **kwargs) -> T:
        """
        Execute forward pass on latent representation for input tokens.
        This method must return tensors of corresponding DNN framework.
        This method must be able to record gradients, as it is used internally by gradient based XAI methods.
        """
        raise NotImplementedError


class TextClassifier(ABC):
    """
    An interface for model, trained for text-classification task (aka sentiment analysis).
    TextClassifier is a model with signature F: List['text'] -> np.ndarray
    """

    @abstractmethod
    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        """
        Execute forward pass with plain text inputs, return logits as np.ndarray.
        This method must be able to handle huge batches, as it potentially could be called with entire dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_lookup(self, input_ids):
        """Convert vocabulary ids to model's latent representations"""
        raise NotImplementedError

    def get_embeddings(self, x_batch: List[str]) -> Tuple[np.ndarray, Dict]:
        from quantus.helpers.utils_nlp import safe_as_array

        """Do batch encode, unpack input ids, convert to embeddings."""
        input_ids, predict_kwargs = self.tokenizer.get_input_ids(x_batch)
        return safe_as_array(self.embedding_lookup(input_ids)), predict_kwargs

    @property
    @abstractmethod
    def tokenizer(self) -> Tokenizable:
        pass


# ---------- QA, NLI, Text Generation, Summarization, NER and more models to follow ----------
# Or actually no, Quantus is designed for classifiers, so I probably will just show examples of other tasks separately.
