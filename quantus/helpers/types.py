# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    overload,
)

import numpy as np
from numpy.typing import ArrayLike

from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import TextClassifier, Tokenizable

if TYPE_CHECKING:
    import torch.nn as nn
    from tensorflow import keras
    from transformers import PreTrainedTokenizerBase

    ModelT = Union[keras.Model, nn.Module, ModelInterface, TextClassifier]
    TokenizerT = Union[PreTrainedTokenizerBase, Tokenizable]

Explanation = Tuple[List[str], np.ndarray]
MetricScores = Union[np.ndarray, float, Dict[str, Union[np.ndarray, float]]]


class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        ...


class PerturbFn(Protocol):
    @overload
    def __call__(self, a: List[str], **kwargs) -> List[str]:
        ...

    @overload
    def __call__(self, a: SupportsArray, **kwargs) -> np.ndarray:
        ...

    def __call__(self, a, **kwargs):
        ...


class SimilarityFn(Protocol):
    def __call__(self, a: SupportsArray, b: SupportsArray, **kwargs) -> ArrayLike:
        ...


class NormFn(Protocol):
    def __call__(self, a: SupportsArray, **kwargs) -> ArrayLike:
        ...


class NormaliseFn(Protocol):
    def __call__(self, a: SupportsArray, **kwargs) -> np.ndarray:
        ...


class ExplainFn(Protocol):
    @overload
    def __call__(
        self, model: TextClassifier, x_batch: np.ndarray, y_batch: np.ndarray, **kwargs
    ) -> np.ndarray:
        ...

    @overload
    def __call__(
        self, model: TextClassifier, x_batch: List[str], y_batch: np.ndarray, **kwargs
    ) -> List[Explanation]:
        ...

    @overload
    def __call__(
        self, model, x_batch: SupportsArray, y_batch: np.ndarray, **kwargs
    ) -> np.ndarray:
        ...

    def __call__(self, model: ModelT, x_batch, y_batch: np.ndarray, **kwargs):
        ...


class AggregateFn(Protocol):
    def __call__(self, a: SupportsArray, **kwargs) -> ArrayLike:
        ...


class PersistFn(Protocol):
    @overload
    def __call__(
        self, metric_name: str, explain_fn_kwargs: Dict[str, ...], scores: np.ndarray
    ) -> None:
        ...

    @overload
    def __call__(
        self,
        metric_name: str,
        explain_fn_kwargs: Dict[str, ...],
        scores: Dict[str, np.ndarray],
    ) -> None:
        ...

    def __call__(self, metric_name, explain_fn_kwargs, scores) -> None:
        ...


CallKwargs = TypedDict(
    "CallKwargs",
    dict(
        model=Any,
        x_batch=np.ndarray | List[str],
        y_batch=Optional[np.ndarray],
        a_batch=Optional[np.ndarray | List[Explanation]],
        channel_first=Optional[bool],
        explain_func=ExplainFn,
        explain_func_kwargs=Optional[Dict[str, ...]],
        model_predict_kwargs=Optional[Dict[str, ...]],
        softmax=Optional[bool],
        device=Optional[str],
        batch_size=int,
        custom_batch=Optional[Any],
        s_batch=Optional[Any],
        tokenizer=Optional[Any],
    ),
    total=False,
)
