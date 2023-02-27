from __future__ import annotations

from enum import Enum, auto
from typing import (
    List,
    Tuple,
    Callable,
    Union,
    Optional,
)
import numpy as np
from quantus.nlp.helpers.model.text_classifier import TextClassifier


Explanation = Tuple[List[str], np.ndarray]
PerturbFn = Union[Callable[[List[str]], List[str]], Callable[[np.ndarray], np.ndarray]]

ExplainFn = Union[
    Callable[[TextClassifier, List[str], np.ndarray], List[Explanation]],
    Callable[
        [TextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray
    ],
]

NormaliseFn = Callable[[np.ndarray], np.ndarray]
SimilarityFn = Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
NormFn = Callable[[np.ndarray], Union[float, np.ndarray]]


class PerturbationType(Enum):
    plain_text = auto()
    latent_space = auto()
