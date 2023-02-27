from __future__ import annotations

from abc import ABC
from typing import Optional, Dict

from quantus.nlp.helpers.utils import value_or_default
from quantus.nlp.helpers.types import PerturbFn
from quantus.nlp.metrics.batched_metric import BatchedMetric


class BatchedPerturbationMetric(BatchedMetric, ABC):
    def __init__(
        self,
        perturb_func: PerturbFn,
        perturb_func_kwargs: Optional[Dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
