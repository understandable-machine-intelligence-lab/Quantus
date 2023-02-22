from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict

from transformers import PreTrainedTokenizerBase
from quantus.nlp.helpers.model.text_classifier import Tokenizer
from quantus.nlp.helpers.utils import value_or_default


class HuggingFaceTokenizer(Tokenizer):

    """A wrapper around HuggingFace's hub tokenizers, which encapsulates common functionality used in Quantus."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def tokenize(self, text: List[str]) -> Dict[str, np.ndarray]:
        return self.tokenizer(text, padding="longest", return_tensors="np").data

    def convert_ids_to_tokens(self, ids: np.ndarray) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def join_tokens(
        self, tokens: List[List[str]], ignore_special_tokens: Optional[List[str]] = None
    ) -> List[str]:
        ignore_special_tokens = value_or_default(ignore_special_tokens, lambda: [])
        vocab = defaultdict(lambda: self.tokenizer.unk_token_id)  # type: ignore
        vocab.update(self.tokenizer.get_vocab())
        special_tokens = list(self.tokenizer.special_tokens_map.values())
        ids_batch = []
        for sentence in tokens:
            ids = []
            for word in sentence:
                if word in special_tokens and word not in ignore_special_tokens:
                    continue
                ids.append(vocab[word])
            ids_batch.append(ids)
        return self.tokenizer.batch_decode(ids_batch)
