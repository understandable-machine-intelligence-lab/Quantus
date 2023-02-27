from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING, Generator, Union
from copy import deepcopy
from multimethod import multimethod
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from quantus.nlp.helpers.utils import (
    batch_list,
    value_or_default,
    map_dict,
)
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TensorLike  # pragma: not covered


class TorchHuggingFaceTextClassifier(TextClassifier):
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

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def device(self):
        return self._device

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self._model.load_state_dict(weights)

    def embedding_lookup(self, input_ids: TensorLike, **kwargs) -> torch.Tensor:
        input_ids = self._to_tensor(input_ids)
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
        inputs_embeds = self._to_tensor(inputs_embeds, dtype=self._model.dtype)
        kwargs = map_dict(kwargs, lambda x: self._to_tensor(x))
        return self._model(None, inputs_embeds=inputs_embeds, **kwargs).logits

    def predict(self, text: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        if len(text) <= batch_size:
            return self._predict_on_batch(text)

        batched_text = batch_list(text, batch_size)
        logits = []

        for i in batched_text:
            logits.append(self._predict_on_batch(i))

        return np.vstack(logits)

    def _predict_on_batch(self, text: List[str]) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(text)
        encoded_inputs = map_dict(encoded_inputs, lambda x: self._to_tensor(x))
        logits = self._model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator:
        original_parameters = self._model.state_dict()
        random_layer_model = deepcopy(self._model)
        random_layer_model_wrapper = TorchHuggingFaceTextClassifier(
            self._tokenizer._tokenizer,  # noqa
            random_layer_model,
            self._device,
        )

        modules = [
            layer
            for layer in self._model.named_modules()
            if (hasattr(layer[1], "reset_parameters"))
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                random_layer_model.load_state_dict(original_parameters)
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()

            yield module[0], random_layer_model_wrapper

    def _to_tensor(self, x: TensorLike, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self._device, **kwargs)

    @multimethod
    def get_hidden_representations(self, x_batch, **kwargs) -> np.ndarray:
        pass

    @get_hidden_representations.register
    def get_hidden_representations(self, x_batch: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(x_batch)
        encoded_inputs = map_dict(encoded_inputs, lambda x: self._to_tensor(x))
        embeddings = self._to_tensor(
            self.embedding_lookup(encoded_inputs.pop("input_ids")),
            dtype=self._model.dtype,
        )
        return self.get_hidden_representations(embeddings, **encoded_inputs)

    @get_hidden_representations.register
    def get_hidden_representations(self, x_batch: np.ndarray, **kwargs) -> np.ndarray:
        x_batch = self._to_tensor(x_batch, dtype=self._model.dtype)
        return self.get_hidden_representations(x_batch, **kwargs)

    @get_hidden_representations.register
    def get_hidden_representations(self, x_batch: torch.Tensor, **kwargs) -> np.ndarray:
        predict_kwargs = map_dict(kwargs, lambda x: self._to_tensor(x))
        hidden_states = self._model(
            None, inputs_embeds=x_batch, output_hidden_states=True, **predict_kwargs
        ).hidden_states
        hidden_states = [i.detach().cpu().numpy() for i in hidden_states]
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states
