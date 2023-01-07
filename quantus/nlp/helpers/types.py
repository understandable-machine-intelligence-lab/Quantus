from __future__ import annotations

from typing import List, Tuple, Callable
import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier


PerturbFn = Callable[[List[str], ...], List[str]]
Explanation = Tuple[List[str], np.ndarray]
ExplainFn = Callable[[List[str], np.ndarray, TextClassifier, ...], List[Explanation]]
SimilarityFn = Callable[
    [Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]], np.ndarray
]
NormaliseFn = Callable[[List[Explanation]], List[Explanation]]
PlotFn = Callable[[List[Explanation]], ...]
