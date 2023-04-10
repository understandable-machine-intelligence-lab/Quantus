# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase

from quantus.helpers.collection_utils import add_default_items
from quantus.helpers.model.text_classifier import Tokenizable


class HuggingFaceTokenizer(Tokenizable):
    """A wrapper around HuggingFace's hub tokenizers, which encapsulates common functionality used in Quantus."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def batch_encode(self, text: List[str], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
        kwargs = add_default_items(kwargs, dict(padding="longest", return_tensors="np"))
        return self.tokenizer(text, **kwargs).data

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(list(ids))

    def token_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def batch_decode(
        self, ids: Sequence[int] | Sequence[Sequence[int]] | np.ndarray, **kwargs
    ) -> List[str]:
        return self.tokenizer.batch_decode(ids, **kwargs)

    def split_into_tokens(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def join_tokens(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
