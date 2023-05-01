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
    if value is None:
        return default_factory()
    else:
        return value


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


def _create_div(
    explanation: Explanation,
    label: str,
    ignore_special_tokens: bool,
    special_tokens: List[str],
):
    # Create a container, which inherits root styles.
    div_template = """
        <div class="container">
            <p>
                {{label}} <br>
                {{saliency_map}}
            </p>
        </div>
        """

    # For each token, create a separate highlight span with different background color.
    token_span_template = """
        <span class="highlight-container" style="background:{{color}};">
            <span class="highlight"> {{token}} </span>
        </span>
        """
    tokens = explanation[0]
    scores = explanation[1]
    body = ""
    color_mapper = ColorMapper(np.max(scores), np.min(scores))

    for token, score in zip(tokens, scores):
        if ignore_special_tokens and token in special_tokens:
            continue
        red, green, blue = color_mapper.to_rgb(score)
        token_span = token_span_template.replace(
            "{{color}}", f"rgb({red},{green},{blue})"
        )
        token_span = token_span.replace("{{token}}", token)
        body += token_span + " "

    return div_template.replace("{{label}}", label).replace("{{saliency_map}}", body)


def visualise_explanations_as_html(
    explanations: List[Explanation],
    *,
    labels: Optional[List[str]] = None,
    ignore_special_tokens: bool = False,
    special_tokens: Optional[List[str]] = None,
) -> str:
    """
    Creates a heatmap visualisation from list of explanations. This method should be preferred for longer
    examples. It is rendered correctly in VSCode, PyCharm, Colab, however not in GitHub or JupyterLab.

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.
    labels:
        Optional, list of labels to display on top of each explanation.
    ignore_special_tokens:
        If True, special tokens will not be rendered in heatmap.
    special_tokens:
        List of special tokens to ignore during heatmap creation, default= ["[CLS]", "[END]", "[PAD]"].

    Returns
    -------

    html:
        string containing raw html to visualise explanations.

    Examples
    -------

    >>> from IPython.core.display import HTML
    >>> from quantus.helpers.plotting import visualise_explanations_as_html
    >>> a_batch = ...
    >>> raw_html = visualise_explanations_as_html(a_batch)
    >>> HTML(raw_html)

    """

    special_tokens = value_or_default(special_tokens, lambda: DEFAULT_SPECIAL_TOKENS)
    # Define top-level styles
    heatmap_template = """
        <style>

            .container {
                line-height: 1.4;
                text-align: center;
                margin: 10px 10px 10px 10px;
                color: black;
                background: white;
            }

            p {
                font-size: 16px;
            }

            .highlight-container, .highlight {
                position: relative;
                border-radius: 10% 10% 10% 10%;
            }

            .highlight-container {
                display: inline-block;
            }

            .highlight-container:before {
                content: " ";
                display: block;
                height: 90%;
                width: 100%;
                margin-left: -3px;
                margin-right: -3px;
                position: absolute;
                top: -1px;
                left: -1px;
                padding: 10px 3px 3px 10px;
            }

        </style>

        {{body}}
        """

    spans = ""
    # For each token, create a separate div holding whole input sequence on 1 line.
    for i, explanation in enumerate(explanations):
        label = labels[i] if labels is not None else ""
        div = _create_div(explanation, label, ignore_special_tokens, special_tokens)
        spans += div
    return heatmap_template.replace("{{body}}", spans)


def value_at_index_or_default(values, index, default):
    if len(values) > index:
        return values[index]
    else:
        return default
