# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatchmethod
from typing import List, Optional

import numpy as np
import torch
from transformers import (
    PreTrainedModel,
)

from quantus.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.helpers.model.model_interface import HiddenRepresentationsModel
from quantus.helpers.model.pytorch_model import TorchModelRandomizer
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.utils import map_dict, value_or_default


class TorchHuggingFaceTextClassifier(
    TextClassifier, TorchModelRandomizer, HiddenRepresentationsModel
):
    def __init__(
        self,
        model: PreTrainedModel,
        softmax=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        handle = model.config._name_or_path  # noqa
        self._tokenizer = HuggingFaceTokenizer(handle)
        self.device = value_or_default(device, lambda: torch.device("cpu"))
        self.model = model.to(self.device)

    def embedding_lookup(self, input_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.model.get_input_embeddings()(self.to_tensor(input_ids))

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = self.to_tensor(inputs_embeds, dtype=self.model.dtype)
        kwargs = map_dict(kwargs, self.to_tensor)
        return self.model(inputs_embeds=inputs_embeds, **kwargs).logits

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self._tokenizer.batch_encode(text)
        encoded_inputs = map_dict(encoded_inputs, self.to_tensor)
        logits = self.model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    @singledispatchmethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:  # type: ignore
        encoded_inputs = self._tokenizer.batch_encode(x_batch)
        embeddings = self.to_tensor(
            self.embedding_lookup(encoded_inputs.pop("input_ids")),
            dtype=self.model.dtype,
        )
        return self.get_hidden_representations(embeddings, **encoded_inputs)

    @get_hidden_representations.register
    def _(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        return self.get_hidden_representations(self.to_tensor(x_batch), **kwargs)

    @get_hidden_representations.register
    def _(
        self,
        x_batch: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> np.ndarray:
        predict_kwargs = map_dict(kwargs, lambda x: self.to_tensor(x))
        hidden_states = self.model(
            None, inputs_embeds=x_batch, output_hidden_states=True, **predict_kwargs
        ).hidden_states
        hidden_states = np.stack([i.detach().cpu().numpy() for i in hidden_states])
        return np.moveaxis(hidden_states, 0, 1)

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        return self._tokenizer
