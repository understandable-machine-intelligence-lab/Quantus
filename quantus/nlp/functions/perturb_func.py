from typing import List

import numpy as np
from nlpaug.augmenter.char import KeyboardAug
from nlpaug.augmenter.word import SpellingAug, SynonymAug


def spelling_replacement(x_batch: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by alternative spelling.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> spelling_replacement(x)
    ... ['uneasy mishmash of stiles and genres.']

    """
    aug = SpellingAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(x_batch)


def synonym_replacement(x_batch: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by synonym.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> synonym_replacement(x)
    ... ['nervous mishmash of styles and genres.']
    """
    aug = SynonymAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(x_batch)


def typo_replacement(x_batch: List[str], k: int = 1, **kwargs) -> List[str]:
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
    return aug.augment(x_batch)


def uniform_noise(
    x_batch: np.ndarray, seed: int = 42, noise_type: str = "additive", **kwargs
) -> np.ndarray:
    """Apply uniform noise to arr."""
    noise = np.random.default_rng(seed).uniform(size=x_batch.shape, **kwargs)
    return _apply_noise(x_batch, noise, noise_type)


def gaussian_noise(
    x_batch: np.ndarray, seed: int = 42, noise_type: str = "additive", **kwargs
) -> np.ndarray:
    """Apply gaussian noise to arr."""
    noise = np.random.default_rng(seed).normal(size=x_batch.shape, **kwargs)
    return _apply_noise(x_batch, noise, noise_type)


def _apply_noise(arr: np.ndarray, noise: np.ndarray, noise_type: str) -> np.ndarray:
    if noise_type not in ("additive", "multiplicative"):
        raise ValueError(
            f"Unsupported noise_type, supported are: additive, multiplicative."
        )
    if noise_type == "additive":
        return arr + noise
    if noise_type == "multiplicative":
        return arr * noise
