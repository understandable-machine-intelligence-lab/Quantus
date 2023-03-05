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
