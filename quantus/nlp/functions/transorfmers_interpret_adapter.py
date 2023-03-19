from typing import Tuple, Union, List, Dict

import numpy as np
import torch
from torch import Tensor
from operator import itemgetter
from transformers_interpret import SequenceClassificationExplainer
from transformers import PreTrainedModel, PreTrainedTokenizer

from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import map_optional
from quantus.nlp.helpers.model.torch_hf_text_classifier import (
    TorchHuggingFaceTextClassifier,
)


class IntGradAdapter(SequenceClassificationExplainer):
    def __init__(
        self,
        x_batch: List[str],
        encoded_inputs: Dict,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(model, tokenizer)
        self.x_batch = list(map(self._clean_text, x_batch))
        self.encoded_inputs = encoded_inputs

    def _make_input_reference_pair(
        self, text: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self._batch_index = self.x_batch.index(text)

        input_ids = self.encoded_inputs["input_ids"][self._batch_index]
        text_ids_len = len(input_ids) - 2
        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * text_ids_len
            + [self.sep_token_id]
        )

        return (
            torch.unsqueeze(input_ids, dim=0),
            torch.tensor([ref_input_ids], device=self.device),
            text_ids_len,
        )

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return map_optional(
            self.encoded_inputs.get("attention_mask"),
            lambda x: torch.unsqueeze(x[self._batch_index], dim=0),
        )

    @staticmethod
    def explain_batch(
        model: TorchHuggingFaceTextClassifier,
        x_batch: List[str],
        y_batch: Tensor,
        num_steps: int,
    ) -> List[Explanation]:
        encoded_inputs = model.batch_encode(x_batch)

        explainer = IntGradAdapter(
            x_batch, encoded_inputs, model.unwrap(), model.unwrap_tokenizer()
        )

        a_batch = []
        for x, y in zip(x_batch, y_batch):
            a = explainer(x, y, n_steps=num_steps)
            tokens = list(map(itemgetter(0), a))
            scores = np.asarray(list(map(itemgetter(1), a)))
            a_batch.append((tokens, scores))

        return a_batch


class IntGradEmbeddingsAdapter(SequenceClassificationExplainer):
    @staticmethod
    def explain_batch(
        model: TorchHuggingFaceTextClassifier,
        x_batch: Tensor,
        y_batch: Tensor,
        num_steps: int,
        **kwargs
    ) -> np.ndarray:
        pass
