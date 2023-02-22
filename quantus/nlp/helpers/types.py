from __future__ import annotations

from typing import (
    List,
    Tuple,
    Callable,
    Union,
    TYPE_CHECKING,
    Optional,
)
import numpy as np
from enum import Enum, auto
from quantus.nlp.helpers.model.text_classifier import TextClassifier

if TYPE_CHECKING:
    import tensorflow  # pragma: not covered
    import torch  # pragma: not covered

    TF_TensorLike = Union[tensorflow.Tensor, np.ndarray]  # pragma: not covered
    TensorLike = Union[torch.Tensor, np.ndarray]  # pragma: not covered


PlainTextPerturbFn = Callable[[List[str]], List[str]]
NumericalPerturbFn = Callable[[np.ndarray], np.ndarray]
Explanation = Tuple[List[str], np.ndarray]
ExplainFn = Callable[[TextClassifier, List[str], np.ndarray], List[Explanation]]
NumericalExplainFn = Callable[
    [TextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
]
NormaliseFn = Callable[[np.ndarray], np.ndarray]
SimilarityFn = Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
NormFm = Callable[[np.ndarray], float]


class PerturbationType(Enum):
    """Enum, which represents choice of perturbation strategy."""

    plain_text = auto()
    latent_space = auto()


class NoiseType(Enum):
    """Enum, which represent (numerical) noise application strategy."""

    additive = auto()
    multiplicative = auto()
