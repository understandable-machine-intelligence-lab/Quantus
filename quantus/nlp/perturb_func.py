from __future__ import annotations

from typing import List
from nlpaug.augmenter.word import SynonymAug, SpellingAug


def spelling_replacement(text: List[str], k: int) -> List[str]:
    aug = SpellingAug(aug_max=k, aug_min=k)
    return aug.augment(text)


def replace_synonym(text: List[str], k: int) -> List[str]:
    aug = SynonymAug(aug_max=k, aug_min=k)
    return aug.augment(text)
