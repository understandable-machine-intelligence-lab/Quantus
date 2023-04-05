from __future__ import annotations

from operator import itemgetter
from typing import List, Callable, TypeVar

import keras
import numpy as np
from cachetools import cached
from quantus.helpers.types import Explanation, Explanations

T = TypeVar("T")
R = TypeVar("R")


def map_explanations(
        a_batch: Explanations, fn: Callable[[T], R]
) -> List[R]:
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

    if type_annotation == "numpy.ndarray" or type_annotation == np.ndarray:
        return False
    if type_annotation == "typing.List[str]" or type_annotation == List[str]:
        return True

    raise ValueError(
        f"Unsupported type annotation for perturbation function: {type_annotation}."
    )


def is_tf_model(model):
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return False
    if isinstance(model, keras.Model):
        return True

    try:
        from quantus.helpers.model.tf_hf_model import TFHuggingFaceTextClassifier
        if isinstance(model, TFHuggingFaceTextClassifier):
            return True
    except ModuleNotFoundError:
        return False

    if hasattr(model, "model"):
        return isinstance(model.model, tf.keras.Model)
    return False


def is_torch_model(model):
    try:
        import torch.nn as nn
    except ModuleNotFoundError:
        return False
    if isinstance(model, nn.Module):
        return True
    try:
        from quantus.helpers.model.torch_hf_model import TorchHuggingFaceTextClassifier
        if isinstance(model, TorchHuggingFaceTextClassifier):
            return True
    except ModuleNotFoundError:
        return False
    if hasattr(model, "model"):
        return isinstance(model.model, nn.Module)
    return False
