from __future__ import annotations

import numpy as np
import sys

import tensorflow.python.framework.ops

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    HuggingFaceTextClassifierTF,
)
from transformers import TFPreTrainedModel
from typing import List, Tuple, Dict, Callable, Any, TypeVar


T = TypeVar("T")


def exponential_kernel(distance: float, kernel_width: float = 25) -> np.ndarray:
    """The exponential kernel."""
    return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def normalize_scores(scores: np.ndarray, make_positive: bool = False) -> np.ndarray:
    """Makes the absolute values sum to 1, optionally making them all positive."""
    if len(scores.shape) == 2:
        return np.asarray([normalize_scores(i) for i in scores])
    if len(scores) < 1:
        return scores
    scores = scores + np.finfo(np.float32).eps
    if make_positive:
        scores = np.abs(scores)
    return scores / np.abs(scores).sum(-1)


def wrap_model(model, tokenizer, model_init_kwargs: Dict) -> TextClassifier:
    if isinstance(model, TextClassifier):
        return model
    if isinstance(model, TFPreTrainedModel):
        return HuggingFaceTextClassifierTF(model, tokenizer, **model_init_kwargs)
    if isinstance(model, str):
        return HuggingFaceTextClassifierTF.from_pretrained(model)
    raise NotImplementedError()


def value_or_default(value: T, default: Callable[[], T] | T) -> T:
    if value is not None:
        return value
    if isinstance(default, Callable):
        return default()
    return default


def pad_ragged_vectors(
    a: List[np.ndarray],
    b: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    max_len = 0
    for a_i, b_i in zip(a, b):
        max_len = max([max_len, len(a_i), len(b_i)])

    a_padded = []
    b_padded = []
    for a_i, b_i in zip(a, b):
        a_padded.append(_pad_array(a_i, max_len))
        b_padded.append(_pad_array(b_i, max_len))

    return a_padded, b_padded


def pad_ragged_vector(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    max_len = max([len(a), len(b)])
    return _pad_array(a, max_len), _pad_array(b, max_len)


def _pad_array(a: np.ndarray, target_length: int) -> np.ndarray:
    if len(a) == target_length:
        return a
    pad_len = target_length - len(a)
    padding = np.zeros(pad_len)
    return np.concatenate([a, padding])


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = [""]

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def to_numpy(a):
    if isinstance(a, np.ndarray):
        return a
    if safe_isinstance(a, "tensorflow.Tensor"):
        return a.numpy()
    if safe_isinstance(a, "tensorflow.python.framework.ops.EagerTensor"):
        return a.numpy()
    if safe_isinstance(a, "torch.Tensor"):
        return a.cpu().numpy()
