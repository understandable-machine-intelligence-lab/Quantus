# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatchmethod
from typing import Dict, List, Optional, Generator
from functools import lru_cache
from abc import abstractmethod

import numpy as np
import torch
from torch import nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from quantus.helpers.torch_model_randomisation import (
    get_random_layer_generator,
    random_layer_generator_length,
)

from quantus.helpers.utils import map_dict
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.hf_tokenizer import HuggingFaceTokenizer
from quantus.nlp.helpers.utils import value_or_default, add_default_items


class TorchHuggingFaceTextClassifier(HuggingFaceTokenizer, TextClassifier):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: Optional[torch.device],
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._device = value_or_default(device, lambda: torch.device("cpu"))
        self._model = model.to(self._device)

    @staticmethod
    def from_pretrained(
        handle: str, device: Optional[torch.device] = None, **kwargs
    ) -> TorchHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate Torch models from HuggingFace Hub."""

        return TorchHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            AutoModelForSequenceClassification.from_pretrained(handle, **kwargs),
            device,
        )

    def embedding_lookup(self, input_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        input_ids = self.to_tensor(input_ids)
        word_embeddings = self._model.get_input_embeddings()(input_ids)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        try:
            position_embeddings = self._model.get_position_embeddings()(position_ids)
            word_embeddings = word_embeddings + position_embeddings
        except NotImplementedError:
            pass
        return word_embeddings

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = self.to_tensor(inputs_embeds, dtype=self._model.dtype)
        kwargs = map_dict(kwargs, lambda x: self.to_tensor(x))
        return self._model(inputs_embeds=inputs_embeds, **kwargs).logits

    def batch_encode(self, text: List[str], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
        kwargs = add_default_items(kwargs, dict(return_tensors="pt"))
        return super().batch_encode(text, **kwargs)  # type: ignore

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self.batch_encode(text)
        logits = self._model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    @singledispatchmethod
    def get_hidden_representations(
        self,
        x_batch,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> np.ndarray:  # type: ignore
        pass

    @get_hidden_representations.register
    def _(self, x_batch: list, **kwargs) -> np.ndarray:
        encoded_inputs = self.batch_encode(x_batch)
        embeddings = self.to_tensor(
            self.embedding_lookup(encoded_inputs.pop("input_ids")),
            dtype=self._model.dtype,
        )
        return self.get_hidden_representations(embeddings, **encoded_inputs)

    @get_hidden_representations.register
    def _(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        return self.get_hidden_representations(
            self.to_tensor(x_batch, dtype=torch.float32), **kwargs
        )

    @get_hidden_representations.register
    def _(self, x_batch: torch.Tensor, **kwargs) -> np.ndarray:
        predict_kwargs = map_dict(kwargs, lambda x: self.to_tensor(x))
        hidden_states = self._model(
            None, inputs_embeds=x_batch, output_hidden_states=True, **predict_kwargs
        ).hidden_states
        hidden_states = np.stack([i.detach().cpu().numpy() for i in hidden_states])
        return np.moveaxis(hidden_states, 0, 1)

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator:
        og_weights = self.weights.copy()
        generator = get_random_layer_generator(self.unwrap(), order, seed)
        for name, model in generator:
            self.weights = model.state_dict()
            yield name, self
        self.weights = og_weights

    def to_tensor(self, x: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, **kwargs)

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self._model.load_state_dict(weights)

    @property
    def random_layer_generator_length(self) -> int:
        return random_layer_generator_length(self.unwrap())

    def unwrap_tokenizer(self):
        return self._tokenizer

    def unwrap(self) -> nn.Module:
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def embeddings_dtype(self):
        @lru_cache(maxsize=None)
        def _embeddings_dtype():
            return self.embedding_lookup(np.ones(shape=(1, 1))).dtype

        return _embeddings_dtype()
