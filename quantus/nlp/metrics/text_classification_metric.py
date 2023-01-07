from abc import ABC, abstractmethod

import numpy as np
from typing import List, Optional, Dict
from quantus.metrics.base_batched import BatchedPerturbationMetric


from quantus.nlp.helpers.types import ExplainFn
from quantus.nlp.helpers.model.text_classifier import TextClassifier


class BatchedTextClassificationMetric(BatchedPerturbationMetric):
    def general_preprocess(
        self,
        *,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        explain_func: ExplainFn,
        explain_func_kwargs: Optional[Dict],
        model_predict_kwargs: Optional[Dict],
        softmax: bool,
        device: Optional[str],
        custom_batch: Optional[np.ndarray],
        **kwargs
    ) -> Dict:
        pass
