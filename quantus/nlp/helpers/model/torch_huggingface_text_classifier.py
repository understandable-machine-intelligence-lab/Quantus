from __future__ import annotations

import numpy as np
from typing import List, Optional
from multimethod import multimethod
from torch import nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from quantus.nlp.helpers.utils import (
    value_or_default,
    map_dict,
)
from quantus.nlp.helpers.model.torch_text_classifier import TorchTextClassifier
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer


class TorchHuggingFaceTextClassifier(TorchTextClassifier):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: Optional[torch.device],
    ):
        self._device = value_or_default(device, lambda: torch.device("cpu"))
        self._model = model.to(self._device)
        self._tokenizer = HuggingFaceTokenizer(tokenizer)

    @staticmethod
    def from_pretrained(
        handle: str,
        device: Optional[torch.device] = None,
    ) -> TorchHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate Torch models from HuggingFace Hub."""

        return TorchHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle, add_prefix_space=True),
            AutoModelForSequenceClassification.from_pretrained(handle),
            device,
        )

    def embedding_lookup(
        self, input_ids: torch.Tensor | np.ndarray, **kwargs
    ) -> torch.Tensor:
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
        return self._model(None, inputs_embeds=inputs_embeds, **kwargs).logits

    def predict_on_batch(self, text: List[str]) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(text)
        encoded_inputs = map_dict(encoded_inputs, lambda x: self.to_tensor(x))
        logits = self._model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    @multimethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:
        pass

    @get_hidden_representations.register
    def _(self, x_batch: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(x_batch)
        encoded_inputs = map_dict(encoded_inputs, lambda x: self.to_tensor(x))
        embeddings = self.to_tensor(
            self.embedding_lookup(encoded_inputs.pop("input_ids")),
            dtype=self._model.dtype,
        )
        return self.get_hidden_representations(embeddings, **encoded_inputs)

    @get_hidden_representations.register
    def _(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        x_batch = self.to_tensor(x_batch, dtype=self._model.dtype)
        return self.get_hidden_representations(x_batch, **kwargs)

    @get_hidden_representations.register
    def _(self, x_batch: torch.Tensor, **kwargs) -> np.ndarray:
        predict_kwargs = map_dict(kwargs, lambda x: self.to_tensor(x))
        hidden_states = self._model(
            None, inputs_embeds=x_batch, output_hidden_states=True, **predict_kwargs
        ).hidden_states
        hidden_states = [i.detach().cpu().numpy() for i in hidden_states]
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def internal_model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def clone(self) -> TorchTextClassifier:
        return self.from_pretrained(self._model.name_or_path, device=self._device)
