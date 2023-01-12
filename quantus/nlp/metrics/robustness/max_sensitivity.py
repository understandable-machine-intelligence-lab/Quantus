from __future__ import annotations

from quantus.nlp.metrics.text_classification_metric import (
    BatchedTextClassificationMetric,
)


class MaxSensitivity(BatchedTextClassificationMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass

    def evaluate_batch(*args, **kwargs):
        pass
