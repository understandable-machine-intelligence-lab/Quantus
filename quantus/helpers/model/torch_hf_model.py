# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatchmethod, partial, cached_property
from typing import List, Generator, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from quantus.helpers.collection_utils import map_dict, value_or_default
from quantus.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.torch_utils import random_layer_generator, list_layers


class TorchHuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: HuggingFaceTokenizer,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = value_or_default(device, lambda: torch.device("cpu"))
        self.model = model.to(self.device)

    def embedding_lookup(self, input_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.tensor(input_ids, device=self.device)
        return self.model.get_input_embeddings()(input_ids)

    @singledispatchmethod
    def predict(self, x_batch: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self.tokenizer.batch_encode(x_batch)
        encoded_inputs = map_dict(
            encoded_inputs, partial(torch.tensor, device=self.device)
        )
        logits = self.model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    @predict.register
    def _(self, x_batch: np.ndarray, **kwargs) -> torch.Tensor:
        return self.predict(torch.tensor(x_batch, device=self.device), **kwargs)

    @predict.register
    def _(self, x_batch: torch.Tensor, **kwargs) -> torch.Tensor:
        kwargs = map_dict(kwargs, partial(torch.tensor, device=self.device))
        return self.model(None, inputs_embeds=x_batch, **kwargs).logits

    def get_random_layer_generator(
        self, order = "top_down", seed: int = 42
    ) -> Generator[TorchHuggingFaceTextClassifier, None, None]:
        return random_layer_generator(self, order, seed)

    @cached_property
    def random_layer_generator_length(self) -> int:
        return len(list_layers(self.get_model()))

    def get_model(self) -> nn.Module:
        return self.model

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, original_parameters: Dict[str, torch.Tensor]):
        self.model.load_state_dict(original_parameters)

    @singledispatchmethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:  # type: ignore
        encoded_inputs = self.tokenizer.batch_encode(x_batch)
        embeddings = self.embedding_lookup(encoded_inputs.pop("input_ids"))
        return self.get_hidden_representations(embeddings, **encoded_inputs)

    @get_hidden_representations.register
    def _(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        return self.get_hidden_representations(
            torch.tensor(x_batch, device=self.device), **kwargs
        )

    @get_hidden_representations.register
    def _(self, x_batch: torch.Tensor, **kwargs) -> np.ndarray:
        def map_fn(x):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, device=self.device)
            else:
                return x

        predict_kwargs = map_dict(kwargs, map_fn)
        hidden_states = self.model(
            None, inputs_embeds=x_batch, output_hidden_states=True, **predict_kwargs
        ).hidden_states
        hidden_states = np.stack([i.detach().cpu().numpy() for i in hidden_states])
        return np.moveaxis(hidden_states, 0, 1)
