from __future__ import annotations

from typing import List

import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.helpers.utils import calculate_auc
from quantus.nlp.metrics.batched_metric import BatchedMetric


class TokenFlipping(BatchedMetric):
    """
    References:
        - https://arxiv.org/abs/2202.07304, https://github.com/AmeenAli/XAI_Transformers/blob/main/SST/run_sst.py#L127
    """

    def __init__(
        self,
        *,
        return_auc_per_sample: bool = False,
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_auc_per_sample = return_auc_per_sample
        self.mask_token = mask_token

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
    ) -> np.ndarray | float:
        # Get indices of sorted attributions (descending).
        a_indices = self.argsort_attributions(a_batch)
        n_perturbations = len(a_batch[0])
        n_perturbations = len(range(0, len(a_indices), n_perturbations))
        preds = [None for _ in range(n_perturbations)]

        for i, flip_indices in enumerate(a_indices):
            x_perturbed = self.get_masked_inputs(a_batch, flip_indices, model)
            # Predict on perturbed input x.
            y_pred_perturb = float(model.predict(x_perturbed)[:, y_batch])
            preds[i] = y_pred_perturb

        if self.return_auc_per_sample:
            return calculate_auc(np.asarray(preds))

        return preds

    @property
    def auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean([calculate_auc(np.array(curve)) for curve in self.last_results])

    def get_masked_inputs(
        self, a_batch: List[Explanation], mask_indices: List[str], model: TextClassifier
    ) -> List[str]:
        a_batch_tokens = [i[0] for i in a_batch]

        masked_inputs = []
        for a, i in zip(a_batch_tokens, mask_indices):
            a[i] = self.mask_token
            masked_x = model.tokenizer.join_tokens(a)
            masked_inputs.append(masked_x)

        return masked_inputs

    @staticmethod
    def argsort_attributions(a_batch: List[Explanation]) -> np.ndarray:
        indices = []

        for a in a_batch:
            indices.append(np.argsort(a[1]))

        return np.asarray(indices).T
