from __future__ import annotations
from typing import Optional, Dict

from quantus.nlp.helpers.utils import value_or_default
from quantus.nlp.helpers.types import (
    PlainTextPerturbFn,
    NumericalPerturbFn,
    PerturbationType,
)
from quantus.nlp.metrics.batched_metric import BatchedMetric


class BatchedPerturbationMetric(BatchedMetric):
    def __init__(
        self,
        perturbation_type: PerturbationType,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn,
        perturb_func_kwargs: Optional[Dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(perturbation_type, PerturbationType):
            raise ValueError("Only enum values of type PerturbationType are allowed")

        self.perturbation_type = perturbation_type
        self.perturb_func = perturb_func
        self.explain_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
