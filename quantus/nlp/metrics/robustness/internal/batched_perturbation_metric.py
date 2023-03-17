# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC
from typing import Dict, Optional

from quantus.nlp.helpers.types import PerturbFn
from quantus.nlp.helpers.utils import value_or_default
from quantus.nlp.metrics.text_classification_metric import TextClassificationMetric


class BatchedPerturbationMetric(TextClassificationMetric, ABC):
    def __init__(
        self,
        perturb_func: PerturbFn,
        perturb_func_kwargs: Optional[Dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
