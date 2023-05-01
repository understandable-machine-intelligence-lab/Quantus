# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

# The source file is named q_types to avoid clash with builtin types
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Protocol,
    Tuple,
    Union,
    overload,
    Callable,
    runtime_checkable,
    Literal,
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
AggregateFn = Callable[[np.ndarray], ArrayLike]
SimilarityFn = Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]
DataDomain = Literal["Image", "Time-Series", "Tabular", "NLP"]
FlipTask = Literal["pruning", "activation"]
LayerOrderT = Literal["independent", "top_down"]


@runtime_checkable
class PerturbFn(Protocol):
    @overload
    def __call__(self, a: list[str], **kwargs) -> list[str]:
        ...

    @overload
    def __call__(self, a: np.ndarray, **kwargs) -> np.ndarray:
        ...

    def __call__(self, a, **kwargs):
        ...


@runtime_checkable
class NormFn(Protocol):
    def __call__(self, a: ArrayLike, **kwargs) -> ArrayLike:
        ...


@runtime_checkable
class NormaliseFn(Protocol):
    def __call__(self, a: np.ndarray, **kwargs) -> np.ndarray:
        ...


@runtime_checkable
class ExplainFn(Protocol):
    @overload
    def __call__(
        self, model: TextClassifier, x_batch: np.ndarray, y_batch: np.ndarray, **kwargs
    ) -> np.ndarray:
        ...

    @overload
    def __call__(
        self, model: TextClassifier, x_batch: list[str], y_batch: np.ndarray, **kwargs
    ) -> list[Explanation]:
        ...

    @overload
    def __call__(
        self, model, x_batch: np.ndarray, y_batch: np.ndarray, **kwargs
    ) -> np.ndarray:
        ...

    def __call__(self, model: ModelT, x_batch, y_batch: np.ndarray, **kwargs):
        ...
