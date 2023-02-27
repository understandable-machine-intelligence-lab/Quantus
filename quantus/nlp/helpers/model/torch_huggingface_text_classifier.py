from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING, Generator
from copy import deepcopy
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
    apply_to_dict,
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

    def embedding_lookup(self, input_ids: TensorLike, **kwargs) -> torch.Tensor:
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
        kwargs = apply_to_dict(kwargs, lambda x: self.to_tensor(x))
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
        encoded_inputs = apply_to_dict(encoded_inputs, lambda x: self.to_tensor(x))
        logits = self._model(**encoded_inputs).logits
        return logits.detach().cpu().numpy()

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self._model.load_state_dict(weights)

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

    def get_hidden_representations(
        self,
        x_batch: List[str],
    ) -> np.ndarray:
        encoded_inputs = self._tokenizer.tokenize(x_batch)
        encoded_inputs = apply_to_dict(encoded_inputs, lambda x: self.to_tensor(x))
        embeddings = self.to_tensor(
            self.embedding_lookup(encoded_inputs.pop("input_ids")),
            dtype=self._model.dtype,
        )
        return self.get_hidden_representations_embeddings(embeddings, **encoded_inputs)

    def get_hidden_representations_embeddings(
        self, x_batch: np.ndarray | torch.Tensor, **kwargs
    ) -> np.ndarray:
        x_batch = self.to_tensor(x_batch, dtype=self._model.dtype)
        predict_kwargs = apply_to_dict(kwargs, lambda x: self.to_tensor(x))

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

    def to_tensor(self, x: TensorLike, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self._device, **kwargs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def device(self):
        return self._device
