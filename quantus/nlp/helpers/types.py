# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier

Explanation = Tuple[List[str], np.ndarray]
PerturbFn = Union[Callable[[List[str]], List[str]], Callable[[np.ndarray], np.ndarray]]

ExplainFn = Union[
    Callable[[TextClassifier, List[str], np.ndarray], List[Explanation]],
    Callable[[TextClassifier, np.ndarray, np.ndarray], np.ndarray],
]

NormaliseFn = Callable[[np.ndarray], np.ndarray]
SimilarityFn = Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]
NormFn = Callable[[np.ndarray], Union[float, np.ndarray]]
