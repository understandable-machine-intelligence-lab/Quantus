from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier

Explanation = Tuple[List[str], np.ndarray]
PerturbFn = Union[Callable[[List[str]], List[str]], Callable[[np.ndarray], np.ndarray]]

ExplainFn = Union[
    Callable[[TextClassifier, List[str], np.ndarray], List[Explanation]],
    Callable[
        [TextClassifier, np.ndarray, np.ndarray], np.ndarray
    ],
]

NormaliseFn = Callable[[np.ndarray], np.ndarray]
SimilarityFn = Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
NormFn = Callable[[np.ndarray], Union[float, np.ndarray]]
