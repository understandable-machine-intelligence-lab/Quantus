from __future__ import annotations

import typing
from functools import lru_cache
from importlib import util
from operator import itemgetter
from typing import List, Callable, TypeVar

import numpy as np
from cachetools import cached
from quantus.helpers.types import Explanation

T = TypeVar("T")
R = TypeVar("R")


def is_transformers_available() -> bool:
    return util.find_spec("transformers") is not None


def map_explanations(a_batch, fn: Callable[[T], R]) -> List[R]:
    """Apply fn to a_batch, supports token-scores tuples as well as raw scores."""
    if isinstance(a_batch, List):
        return [(tokens, fn(scores)) for tokens, scores in a_batch]
    else:
        return fn(a_batch)  # type: ignore


def get_scores(a_batch: List[Explanation]) -> np.ndarray:
    """Get scores out of token + score tuples."""
    # I was just tired having to type it every time.
    return np.asarray(list(map(itemgetter(1), a_batch)))


@cached(cache={}, key=lambda f: f.__name__)
def is_plain_text_perturbation(func: Callable) -> bool:
    """Determine perturbation type based on perturb_func signature."""
    _annotations = func.__annotations__  # noqa
    if "return" in _annotations:
        type_annotation = _annotations["return"]
    elif "x_batch" in _annotations:
        type_annotation = _annotations["x_batch"]
    else:
        raise ValueError(
            f"Could not determine type of perturbation from perturbation functions signature. "
            f"Please add type annotation to `x_batch` argument or add return type annotation."
        )

    return type_annotation in (
        "typing.List[str]",
        "List[str]",
        "list[str]",
        typing.List[str],
        List[str],
        list[str],
    )


def is_nlpaug_available() -> bool:
    return util.find_spec("nlpaug") is not None


def is_nltk_available() -> bool:
    return util.find_spec("nltk") is not None
