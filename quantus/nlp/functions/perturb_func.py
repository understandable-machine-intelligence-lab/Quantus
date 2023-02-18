from typing import List

import numpy as np
from nlpaug.augmenter.word import SynonymAug, SpellingAug
from nlpaug.augmenter.char import KeyboardAug

from quantus.nlp.helpers.utils import apply_noise
from quantus.nlp.helpers.types import NoiseType


def spelling_replacement(text: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by alternative spelling.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> spelling_replacement(x)
    ... ['uneasy mishmash of stiles and genres.']

    """
    aug = SpellingAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(text)


def synonym_replacement(text: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by synonym.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> synonym_replacement(x)
    ... ['nervous mishmash of styles and genres.']
    """
    aug = SynonymAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(text)


def typo_replacement(text: List[str], k: int = 1, **kwargs) -> List[str]:
    """
    Replace k characters in k words in each entry of text mimicking typo.

    Examples
    --------
    >>> x = ["uneasy mishmash of styles and genres."]
    >>> typo_replacement(x)
    ... ['uneasy mishmash of xtyles and genres.']
    """
    aug = KeyboardAug(
        aug_char_max=k, aug_char_min=k, aug_word_min=k, aug_word_max=k, **kwargs
    )
    return aug.augment(text)


def uniform_noise(
    arr: np.ndarray,
    noise_type: NoiseType = NoiseType.additive,
    seed: int = 42,
    **kwargs
) -> np.ndarray:
    """Apply uniform noise to arr."""
    noise = np.random.default_rng(seed).uniform(size=arr.shape, **kwargs)
    return apply_noise(arr, noise, noise_type)


def gaussian_noise(
    arr: np.ndarray,
    noise_type: NoiseType = NoiseType.additive,
    seed: int = 42,
    **kwargs
) -> np.ndarray:
    """Apply gaussian noise to arr."""
    noise = np.random.default_rng(seed).normal(size=arr.shape, **kwargs)
    return apply_noise(arr, noise, noise_type)
