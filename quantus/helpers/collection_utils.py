from __future__ import annotations

from functools import singledispatch
from typing import (
    Any,
    Dict,
    List,
    Iterable,
    TypeVar,
    Callable,
    Sequence,
)

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import gen_batches

from quantus.helpers.tf_utils import is_tensorflow_available
from quantus.helpers.torch_utils import is_torch_available


@singledispatch
def safe_as_array(a: ArrayLike, force: bool = False) -> np.ndarray:
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
    return np.asarray(a)


if is_torch_available():
    import torch

    @safe_as_array.register
    def _(a: torch.Tensor, force: bool = False) -> np.ndarray:
        return a.detach().cpu().numpy()


if is_tensorflow_available():
    import tensorflow as tf

    @safe_as_array.register
    def _(a: tf.Tensor, force: bool = False) -> np.ndarray:
        if force:
            return np.array(tf.identity(a))
        return a  # noqa


T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S", bound=Sequence, covariant=True)


def map_dict(
    dictionary: Dict[str, T],
    value_mapper: Callable[[T], R],
    key_mapper: Callable[[str], str] = lambda x: x,
) -> Dict[str, R]:
    """Applies func to values in dict. Additionally, if provided can also map keys."""
    result = {}
    for k, v in dictionary.items():
        result[key_mapper(k)] = value_mapper(v)
    return result


def flatten(list_2d: Iterable[Iterable[T]]) -> List[T]:
    """Does the same as np.reshape(..., -1), but work also on ragged matrices."""
    return [item for sublist in list_2d for item in sublist]


def batch_inputs(flat_list: S[T], batch_size: int) -> List[S[T]]:
    """Divide list in batches of batch_size, despite the name works also for any Sized and SupportsIndex."""
    indices = list(gen_batches(len(flat_list), batch_size))
    return list(map(lambda i: flat_list[i.start : i.stop], indices))


def map_optional(val: T | None, func: Callable[[T], R]) -> R | None:
    """Apply func to value if not None, otherwise return None."""
    if val is None:
        return None
    return func(val)


def add_default_items(
    dictionary: Dict[str, ...] | None, default_items: Dict[str, ...]
) -> Dict[str, Any]:
    """Add default_items into dictionary if not present."""
    if dictionary is None:
        return default_items.copy()

    copy = dictionary.copy()

    for k, v in default_items.items():
        if k not in copy:
            copy[k] = v

    return copy


def value_or_default(value: T | None, default_factory: Callable[[], T]) -> T:
    """Return value from default_factory() if value is None, otherwise value itself."""
    # Default is provided by callable, because otherwise it will force materialization of both values in memory.
    if value is not None:
        return value
    else:
        return default_factory()


K = TypeVar("K")
V = TypeVar("V")


def filter_dict(
    dictionary: Dict[K, V],
    key_filter: Callable[[K], bool] = lambda a: True,
    value_filter: Callable[[V], bool] = lambda b: True,
) -> Dict[K, V]:
    result = {}

    for k, v in dictionary.items():
        if key_filter(k) and value_filter(v):
            result[k] = v

    return result
