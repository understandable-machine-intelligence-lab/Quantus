from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING, Generator
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from quantus.helpers.model.torch_utils import (
    get_hidden_representations,
    get_random_layer_generator,
)
from quantus.nlp.helpers.utils import (
    unpack_token_ids_and_attention_mask,
    batch_list,
    value_or_default,
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
        device: Optional[torch.device],
    ):
        self.device = value_or_default(device, lambda: torch.device("cpu"))
        self.model = model.to(device)
        self.tokenizer = HuggingFaceTokenizer(tokenizer)

    @staticmethod
    def from_pretrained(
        handle: str,
        device: Optional[str | torch.device] = None,
    ) -> TorchHuggingFaceTextClassifier:
        """A method, which mainly should be used to instantiate Torch models from HuggingFace Hub."""

        return TorchHuggingFaceTextClassifier(
            AutoTokenizer.from_pretrained(handle),
            AutoModelForSequenceClassification.from_pretrained(handle),
            device,
        )

    def embedding_lookup(self, input_ids: TensorLike, **kwargs) -> torch.Tensor:
        input_ids = torch.tensor(input_ids).to(self.device)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        word_embeddings = self.model.get_input_embeddings()(input_ids)
        position_embeddings = self.model.get_position_embeddings()(position_ids)
        return word_embeddings + position_embeddings

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        return self.model(None, attention_mask, inputs_embeds=inputs_embeds).logits

    def predict(self, text: List[str], batch_size: int = 64) -> np.ndarray:
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
        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        logits = self.model(input_ids, attention_mask).logits
        return logits.detach().cpu().numpy()

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self.model.load_state_dict(weights)

    @property
    def nr_layers(self) -> int:
        return len(self.model.modules())

    def get_hidden_representations(
        self,
        x_batch: List[str],
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        pass

    def get_hidden_representations_embeddings(
        self,
        x_batch: np.ndarray,
        attention_mask: Optional[np.ndarray],
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        pass

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator:
        return get_random_layer_generator(self.model, order, seed)
