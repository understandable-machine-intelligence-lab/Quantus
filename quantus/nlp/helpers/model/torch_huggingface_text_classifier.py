from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING, Generator, Tuple
from copy import deepcopy
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import torch.nn as nn
from quantus.nlp.helpers.utils import (
    unpack_token_ids_and_attention_mask,
    batch_list,
    value_or_default,
    choose_torch_device,
)
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import TensorLike


class TorchHuggingFaceTextClassifier(TextClassifier):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        handle: str,
        device: Optional[torch.device],
    ):
        self.device = value_or_default(device, choose_torch_device)
        self.handle = handle
        self.model = model.to(self.device)
        self.tokenizer = HuggingFaceTokenizer(tokenizer)

    @staticmethod
    def from_pretrained(
        handle: str,
        device: Optional[torch.device] = None,
    ) -> TorchHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate Torch models from HuggingFace Hub."""

        return TorchHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            AutoModelForSequenceClassification.from_pretrained(handle),
            handle,
            device,
        )

    def embedding_lookup(self, input_ids: TensorLike, **kwargs) -> torch.Tensor:
        input_ids = torch.tensor(input_ids).to(self.device)
        word_embeddings = self.model.get_input_embeddings()(input_ids)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        try:
            position_embeddings = self.model.get_position_embeddings()(position_ids)
            word_embeddings = word_embeddings + position_embeddings
        except NotImplementedError:
            pass
        return word_embeddings

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, device=self.device)
        if not isinstance(inputs_embeds, torch.Tensor):
            try:
                inputs_embeds = torch.tensor(inputs_embeds, device=self.device)
            except TypeError:
                inputs_embeds = torch.tensor(
                    inputs_embeds, device=self.device, dtype=torch.float32
                )
        return self.model(None, attention_mask, inputs_embeds=inputs_embeds).logits

    def predict(self, text: List[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        if len(text) <= batch_size:
            return self._predict_on_batch(text)

        batched_text = batch_list(text, batch_size)
        logits = []

        for i in batched_text:
            logits.append(self._predict_on_batch(i))

        return np.vstack(logits)

    def _predict_on_batch(self, text: List[str]) -> np.ndarray:
        input_ids, attention_mask = unpack_token_ids_and_attention_mask(
            self.tokenizer.tokenize(text)
        )
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        logits = self.model(input_ids, attention_mask).logits
        return logits.detach().cpu().numpy()

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self.model.load_state_dict(weights)

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42, **kwargs
    ) -> Generator[Tuple[nn.Module, TorchHuggingFaceTextClassifier], None, None]:
        original_parameters = self.model.state_dict()
        random_layer_model = deepcopy(self.model)

        modules = [
            l
            for l in self.named_modules_fn(self.model)
            if (hasattr(l[1], "reset_parameters"))
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                random_layer_model.load_state_dict(original_parameters)
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()
            random_layer_model_wrapper = TorchHuggingFaceTextClassifier(
                AutoTokenizer.from_pretrained(self.handle),
                random_layer_model,
                self.handle,
                self.device,
            )
            yield module[0], random_layer_model_wrapper

    @staticmethod
    def named_modules_fn(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        def is_flat(named_module: Tuple[str, nn.Module]) -> bool:
            module = named_module[1]
            return len(list(module.named_children())) == 0

        modules = list(model.named_modules())

        return list(filter(is_flat, modules))

    def get_hidden_representations(
        self,
        x_batch: List[str],
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        inputs_ids, attention_mask = unpack_token_ids_and_attention_mask(
            self.tokenizer.tokenize(x_batch)
        )
        x_batch_embeddings = self.embedding_lookup(inputs_ids)
        return self.get_hidden_representations_embeddings(
            x_batch_embeddings, attention_mask
        )

    def get_hidden_representations_embeddings(
        self,
        x_batch: np.ndarray | torch.Tensor,
        attention_mask: Optional[np.ndarray],
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        try:
            x_batch = torch.tensor(x_batch, device=self.device)
        except TypeError:
            x_batch = torch.tensor(x_batch, device=self.device, dtype=torch.float32)

        attention_mask = torch.tensor(attention_mask, device=self.device)

        hidden_states = self.model(
            None,
            attention_mask,
            inputs_embeds=x_batch,
            output_hidden_states=True,
        ).hidden_states
        hidden_states = [i.detach().cpu().numpy() for i in hidden_states]
        hidden_states = np.asarray(hidden_states)
        hidden_states = np.moveaxis(hidden_states, 0, 1)
        return hidden_states
