# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatch
from functools import partial

from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from cachetools import cached
from sklearn.utils import gen_batches
from quantus.nlp.config import config
from quantus.nlp.helpers.types import Explanation, TextClassifier

T = TypeVar("T")
R = TypeVar("R")


def flatten_list(list_2d: List[List[T]]) -> List[T]:
    """Does the same as np.reshape(..., -1), but work also on ragged vectors."""
    return [item for sublist in list_2d for item in sublist]


def get_input_ids(
    x_batch: List[str], model: TextClassifier
) -> Tuple[Any, Dict[str, Any]]:
    """Do batch encode, unpack input ids and other forward-pass kwargs."""
    encoded_input = model.batch_encode(x_batch)
    return encoded_input.pop("input_ids"), encoded_input  # type: ignore


def value_or_default(value: Optional[T], default_factory: Callable[[], T]) -> T:
    """Return value from default_factory() if value is None, otherwise value itself."""
    # Default is provided by callable, because otherwise it will force materialization of both values in memory.
    if value is not None:
        return value
    else:
        return default_factory()


def batch_list(flat_list: List[T], batch_size: int) -> List[List[T]]:
    """Divide list in batches of batch_size, despite the name works also for any Sized and SupportsIndex."""
    indices = list(gen_batches(len(flat_list), batch_size))
    return list(map(lambda i: flat_list[i.start : i.stop], indices))


def map_explanations(
    a_batch: List[Explanation] | np.ndarray, fn: Callable[[T], R]
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


def map_optional(val: Optional[T], func: Callable[[T], R]) -> Optional[R]:
    """Apply func to value if not None, otherwise return None."""
    if val is None:
        return None
    return func(val)


def add_default_items(
    dictionary: Optional[Dict[str, Any]], default_items: Dict[str, Any]
) -> Dict[str, Any]:
    """Add default_items into dictionary if not present."""
    if dictionary is None:
        return default_items.copy()

    copy = dictionary.copy()

    for k, v in default_items.items():
        if k not in copy:
            copy[k] = v

    return copy


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


def get_logits_for_labels(logits: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    logits:
        2D array of models (output) logits with shape (batch size, num_classes).
    y_batch:
        1D array of labels with shape (batch size,)

    Returns
    -------

    logits:
        1D array

    """
    # Yes, this is a one-liner, yes this could be done in for-loop, but I've spent 2.5 hours debugging why
    # my scores do not look like expected, so let this be separate function, so I don't have to figure it out
    # the hard way again one more time.
    return logits[np.asarray(list(range(y_batch.shape[0]))), y_batch]


def apply_noise(arr: np.ndarray, noise: np.ndarray, noise_type: str) -> np.ndarray:
    """Apply `noise` to `arr` based on `noise_type.`"""
    if noise_type not in ("additive", "multiplicative"):
        raise ValueError(
            f"Unsupported noise_type, supported are: additive, multiplicative."
        )
    if noise_type == "additive":
        return arr + noise
    if noise_type == "multiplicative":
        return arr * noise


@singledispatch
def safe_as_array(a, force: bool = False) -> np.ndarray:
    """
    Convert DNN frameworks' tensors to numpy arrays. Safe means safe from torch complaining about tensors
    being on other device or attached to graph. So, the only one type we're really interested is torch.Tensor.
    In practise, TF tensors can be passed to numpy functions without any issues, so we can avoid overhead of copying them.

    Parameters
    ----------
    a:
        Pytorch or TF tensor.
    force:
        If set to true, will force conversion of TF tensors to numpy arrays.
        This option should be used, when user needs to modify values inside `a`, since TF tensors are read only.

    Returns
    -------
    a:
        np.ndarray or tf.Tensor, a is tf.Tensor and force=False.

    """
    #
    return a


try:
    import torch

    @safe_as_array.register
    def _(a: torch.Tensor, force=False):
        return a.detach().cpu().numpy()

except ModuleNotFoundError:
    pass


try:
    import tensorflow as tf

    tf_function = partial(
        tf.function,
        reduce_retracing=True,
        jit_compile=config.use_xla,
    )

    @safe_as_array.register
    def _(a: tf.Tensor, force=False):
        if force:
            return np.array(tf.identity(a))
        return a

except ModuleNotFoundError:
    pass
