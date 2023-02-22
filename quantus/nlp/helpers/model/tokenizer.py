from abc import abstractmethod, ABC
from typing import List


class Tokenizer(ABC):

    """An interface for object used to convert plain-text inputs into vocabulary id's and vice-versa"""

    @abstractmethod
    def tokenize(self, text: List[str]):
        """Convert batch of plain-text inputs to vocabulary id's."""
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def convert_ids_to_tokens(self, ids) -> List[str]:
        """Convert batch of vocabulary id's batch to batch of plain-text strings."""
        raise NotImplementedError  # pragma: not covered

    @abstractmethod
    def join_tokens(
        self, tokens: List[List[str]], ignore_special_tokens: List[str]
    ) -> List[str]:
        raise NotImplementedError  # pragma: not covered
