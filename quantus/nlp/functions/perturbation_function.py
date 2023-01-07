from typing import List
from nlpaug.augmenter.word import SynonymAug, SpellingAug


def spelling_replacement(text: List[str], k: int, **kwargs) -> List[str]:
    aug = SpellingAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(text)


def synonym_replacement(text: List[str], k: int, **kwargs) -> List[str]:
    aug = SynonymAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(text)
