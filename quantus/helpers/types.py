# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import List, Tuple, Protocol, overload, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from quantus.helpers.model.text_classifier import TextClassifier

Explanation = Tuple[List[str], np.ndarray]


class PerturbFn(Protocol):
    @overload
    def __call__(self, a: List[str], **kwargs) -> List[str]:
        ...

    def __call__(self, a: ArrayLike, **kwargs) -> np.ndarray:
        ...


class SimilarityFn(Protocol):
    @overload
    def __call__(self, a: ArrayLike, b: ArrayLike, **kwargs) -> np.ndarray:
        ...

    def __call__(self, a: ArrayLike, b: ArrayLike, **kwargs) -> float:
        ...


class NormFn(Protocol):
    def __call__(self, a: ArrayLike, **kwargs) -> ArrayLike:
        ...


class NormaliseFn(Protocol):
    def __call__(self, a: ArrayLike, **kwargs) -> ArrayLike:
        ...


class ExplainFn(Protocol):
    @overload
    def __call__(
        self, model: TextClassifier, x_batch: List[str], y_batch: ArrayLike, **kwargs
    ) -> List[Explanation]:
        ...

    @overload
    def __call__(
        self, model: TextClassifier, x_batch: ArrayLike, y_batch: ArrayLike, **kwargs
    ) -> np.ndarray:
        ...

    def __call__(
        self, model, x_batch: ArrayLike, y_batch: ArrayLike, **kwargs
    ) -> np.ndarray:
        ...


class AggregateFn(Protocol):
    @overload
    def __call__(self, a: ArrayLike, **kwargs) -> float:
        ...

    def __call__(self, a: ArrayLike, **kwargs) -> np.ndarray:
        ...
