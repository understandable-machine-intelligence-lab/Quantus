from __future__ import annotations

import sys
import numpy as np
from typing import List, Tuple, Callable, TypeVar, Optional, Any, Dict
from quantus.nlp.helpers.types import SimilarityFn

from quantus.nlp.helpers.types import Explanation, TextClassifier, PerturbationType

T = TypeVar("T")
R = TypeVar("R")


def get_embeddings(
    x_batch: List[str], model: TextClassifier
) -> Tuple[np.ndarray, Dict]:
    encoded_input = model.tokenizer.tokenize(x_batch)
    if isinstance(encoded_input, Dict):
        input_ids = encoded_input.pop("input_ids")
    else:
        input_ids = encoded_input

    embeddings = safe_as_array(model.embedding_lookup(input_ids))
    if isinstance(encoded_input, Dict):
        return embeddings, encoded_input
    return embeddings, {}


def get_input_ids(x_batch: List[str], model: TextClassifier) -> Tuple[np.ndarray, Dict]:
    encoded_input = model.tokenizer.tokenize(x_batch)
    if isinstance(encoded_input, Dict):
        input_ids = encoded_input.pop("input_ids")
    else:
        input_ids = encoded_input
    if isinstance(encoded_input, Dict):
        return input_ids, encoded_input
    return input_ids, encoded_input


def value_or_default(value: Optional[T], default_factory: Callable[[], T]) -> T:
    """Return value from default_factory() if value is None, otherwise value itself."""
    if value is not None:
        return value
    return default_factory()


def pad_ragged_arrays(
    a: np.ndarray, b: np.ndarray, *, pad_value: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a or b, such that both are of the same length."""
    max_len = max([a.shape, b.shape])
    return (
        _pad_array_right(a, max_len, pad_value),
        _pad_array_right(b, max_len, pad_value),
    )


def batch_list(flat_list: List[T], batch_size: int) -> List[List[T]]:
    """
    Convert list to list where each entry is a list of length batch_size.

    Parameters
    ----------
    flat_list:
        Original list.
    batch_size:
        Length of sublist.

    Returns
    -------

    l2:
        List of lists.

    """
    if len(flat_list) % batch_size == 0:
        return np.asarray(flat_list).reshape((-1, batch_size)).tolist()

    batches = flat_list[: len(flat_list) // batch_size * batch_size]
    batches = np.asarray(batches).reshape((-1, batch_size)).tolist()

    batches.append(flat_list[len(flat_list) // batch_size * batch_size :])
    return batches  # type: ignore


def _pad_array_right(
    a: np.ndarray, target_shape: Tuple, pad_value: float
) -> np.ndarray:
    if a.shape == target_shape:
        return a
    if len(a.shape) == 1:
        pad_len = target_shape[0] - len(a)
        return np.pad(a, (0, pad_len), constant_values=pad_value)
    elif len(a.shape) == 2:
        return _pad_array_2d(a, target_shape, pad_value)
    elif len(a.shape) == 3:
        return _pad_array_3d(a, target_shape, pad_value)
    elif len(a.shape) == 4:
        return np.asarray([_pad_array_3d(i, target_shape[1:], pad_value) for i in a])
    else:
        raise ValueError("This is not expected.")


def _pad_array_2d(a: np.ndarray, target_shape: Tuple[int, int], pad_value: float):
    pad_len = target_shape[0] - len(a)
    pad_shape = target_shape[1]
    if pad_len != 0:
        padding = np.full(shape=(pad_len, pad_shape), fill_value=pad_value)
        return np.concatenate([a, padding], axis=0)
    pad_len = len(a)
    pad_shape = target_shape[1] - a.shape[1]

    padding = np.full(shape=(pad_len, pad_shape), fill_value=pad_value)
    return np.concatenate([a, padding], axis=1)


def _pad_array_3d(a: np.ndarray, target_shape: Tuple[int, int, int], pad_value: float):
    batch_size = target_shape[0]
    pad_len = target_shape[1] - a.shape[1]
    pad_shape = target_shape[2]
    padding = np.full(shape=(batch_size, pad_len, pad_shape), fill_value=pad_value)
    return np.concatenate([a, padding], axis=1)


def map_explanations(a_batch: List[Explanation], fn: Callable[[T], R]) -> List[R]:
    """Take absolute value of numerical component of explanations."""
    return [(tokens, fn(scores)) for tokens, scores in a_batch]


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
        class_path_strs = class_path_str  # type: ignore
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


def safe_as_array(arr: T) -> np.ndarray:
    """Convert Tensorflow or Torch tensors to numpy arrays."""
    if safe_isinstance(arr, "torch.Tensor"):
        return arr.detach().cpu().numpy()  # type: ignore
    return np.asarray(arr)


def map_optional(val: Optional[T], func: Callable[[T], R]) -> Optional[R]:
    if val is None:
        return None
    return func(val)


def explanation_similarity(
    a: Explanation,
    b: Explanation,
    similarity_fn: SimilarityFn,
    padded: bool = True,
    numerical_only: bool = True,
) -> T:
    """Compute similarity between batches of explanations using provided similarity_fn."""
    if not padded:
        raise NotImplementedError
    a_padded, b_padded = pad_ragged_arrays(a[1], b[1])
    if not numerical_only:
        raise NotImplementedError
    return similarity_fn(a_padded, b_padded)


def explanations_batch_similarity(
    a_batch: List[Explanation],
    b_batch: List[Explanation],
    similarity_fn: SimilarityFn,
    padded: bool = True,
    numerical_only: bool = True,
) -> T:
    """Compute similarity between batches of explanations using provided similarity_fn."""
    return np.asarray(
        [
            explanation_similarity(a, b, similarity_fn, padded, numerical_only)
            for a, b in zip(a_batch, b_batch)
        ]
    )


def map_dict(dictionary: Dict[str, T], func: Callable[[T], R]) -> Dict[str, R]:
    result = {}
    for k, v in dictionary.items():
        result[k] = func(v)

    return result


def add_default_items(
    dictionary: Optional[Dict[str, Any]], default_items: Dict[str, Any]
) -> Dict[str, Any]:
    if dictionary is None:
        return default_items

    copy = dictionary.copy()

    for k, v in default_items.items():
        if k not in copy:
            copy[k] = v

    return copy


def determine_perturbation_type(func: Callable) -> PerturbationType:
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
        return PerturbationType.latent_space
    if type_annotation == "typing.List[str]" or type_annotation == List[str]:
        return PerturbationType.plain_text

    raise ValueError(
        f"Unsupported type annotation for perturbation function: {type_annotation}."
    )


def apply_noise(arr: np.ndarray, noise: np.ndarray, noise_type: str) -> np.ndarray:
    if noise_type not in ("additive", "multiplicative"):
        raise ValueError(
            f"Unsupported noise_type, supported are: additive, multiplicative."
        )
    if noise_type == "additive":
        return arr + noise
    if noise_type == "multiplicative":
        return arr * noise


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
