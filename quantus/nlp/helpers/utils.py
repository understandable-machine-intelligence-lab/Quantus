from __future__ import annotations

import sys
import numpy as np
from typing import List, Tuple, Callable, TypeVar, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


from quantus.nlp.helpers.types import (
    Explanation,
    NormaliseFn,
    NoiseType,
)

T = TypeVar("T")
R = TypeVar("R")


def value_or_default(value: T, default_factory: Callable[[], T]) -> T:
    """Return value from default_factory() if value is None, otherwise value itself."""
    if value is not None:
        return value
    return default_factory()


def pad_ragged_vector(
    a: np.ndarray, b: np.ndarray, *, pad_value: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a or b, such that both are of the same length."""
    max_len = max([len(a), len(b)])
    return (
        _pad_array_right(a, max_len, pad_value),
        _pad_array_right(b, max_len, pad_value)
    )


def _pad_array_right(a: np.ndarray, target_length: int, pad_value: float) -> np.ndarray:
    if len(a) == target_length:
        return a
    pad_len = target_length - len(a)
    return np.pad(a, (0, pad_len), constant_values=pad_value)


def batch_list(
    flat_list: List[T], batch_size: int, drop_remainder: bool = False
) -> List[List[T]]:
    """
    Convert list to list where each entry is a list of length batch_size.

    Parameters
    ----------
    flat_list:
        Original list.
    batch_size:
        Length of sublist.
    drop_remainder:
        If true, and list is not divisible into n part of batch_size size, drops the last entry, which has different length.

    Returns
    -------

    l2:
        List of lists.

    """
    if len(flat_list) % batch_size == 0:
        return np.asarray(flat_list).reshape((-1, batch_size)).tolist()

    batches = flat_list[: len(flat_list) // batch_size * batch_size]
    batches = np.asarray(batches).reshape((-1, batch_size)).tolist()
    if drop_remainder:
        return batches

    batches.append(flat_list[len(flat_list) // batch_size * batch_size :])
    return batches


def abs_attributions(a_batch: List[Explanation]) -> List[Explanation]:
    """Take absolute value of numerical component of explanations."""
    return [(tokens, np.abs(scores)) for tokens, scores in a_batch]


def normalise_attributions(
    a_batch: List[Explanation], normalise_fn: NormaliseFn
) -> List[Explanation]:
    """Apply normalise_fn to numerical component of explanations."""
    return [(tokens, normalise_fn(scores)) for tokens, scores in a_batch]


def unpack_token_ids_and_attention_mask(
    tokens: Dict[str, np.ndarray] | np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Typically, tokenizers from huggingface hub will return Dict with "input_ids" and "attention_mask".
    Both values must be passed into model for correct inference.
    However, e.g., keras_nlp tokenizers don't return explicit attention mask.
    This function's purpose is to avoid handling both (and potentially more) cases in each metric, explanation function, etc.
    """
    if isinstance(tokens, Dict):
        return tokens["input_ids"], tokens["attention_mask"]
    else:
        return tokens, None


def get_interpolated_inputs(
    baseline: np.ndarray, target: np.ndarray, num_steps: int
) -> np.ndarray:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    if num_steps <= 0:
        return np.array([])
    if num_steps == 1:
        return np.array([baseline, target])

    delta = target - baseline
    scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[
        :, np.newaxis, np.newaxis
    ]
    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs


def map_optional(value: Optional[T], func: Callable[[T], R]) -> Optional[R]:
    """Apply func to value, if value is not None"""
    if value is None:
        return None
    return func(value)


def apply_noise(arr: T, noise: T, noise_type: NoiseType) -> T:
    if not isinstance(noise_type, NoiseType):
        raise ValueError("Only instances of NoiseType enum are supported for noise_type kwarg.")
    if noise_type == NoiseType.additive:
        return arr + noise
    if noise_type == NoiseType.multiplicative:
        return arr * noise


def safe_isinstance(obj: Any, class_path_str: str | List[str] | Tuple) -> bool:
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


def choose_torch_device() -> torch.device:
    """Choose torch hardware acceleration if available."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def explanation_similarity(
    a: Explanation,
    b: Explanation,
    similarity_fn: Callable[[np.ndarray, np.ndarray], T],
    padded: bool = True,
    numerical_only: bool = True,
) -> T:
    """Compute similarity between batches of explanations using provided similarity_fn."""
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    if not numerical_only:
        raise NotImplementedError()
    return similarity_fn(a_padded, b_padded)


def safe_asarray(arr: T) -> np.ndarray:
    """Convert Tensorflow or Torch tensors to numpy arrays."""
    if safe_isinstance(arr, "torch.Tensor"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)
