from __future__ import annotations

from typing import List, Optional, Callable, Dict, no_type_check

import numpy as np

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.helpers.utils import calculate_auc
from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.functions.normalise_func import normalize_sum_to_1


class TokenFlipping(BatchedMetric):
    """
    References:
        - https://arxiv.org/abs/2202.07304
        - https://github.com/AmeenAli/XAI_Transformers/blob/main/SST/run_sst.py#L127
    """

    def __init__(
        self,
        *,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable] = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_auc_per_sample: bool = False,
        mask_token: str = "[UNK]",
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
        )
        self.return_auc_per_sample = return_auc_per_sample
        self.mask_token = mask_token

    @no_type_check
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        *args,
        **kwargs,
    ) -> np.ndarray | float:
        # Get indices of sorted attributions (descending).
        scores = np.full(shape=(len(a_batch[0][1]), len(x_batch)), fill_value=np.NINF)

        mask_indices_batch = []
        for a in a_batch:
            mask_indices_batch.append(np.argsort(a[1])[::-1])
        mask_indices_batch = np.asarray(mask_indices_batch).T

        for i, mask_indices in enumerate(mask_indices_batch):
            a_batch_tokens = [i[0] for i in a_batch]
            x_masked_batch = []
            for a, i in zip(a_batch_tokens, mask_indices):  # type: ignore
                x_masked = a.copy()  # type: ignore
                x_masked[i] = self.mask_token
                x_masked_batch.append(x_masked)

            x_masked_batch = model.tokenizer.join_tokens(
                x_masked_batch, ignore_special_tokens=[self.mask_token]
            )
            # Predict on perturbed input x.
            logits = model.predict(x_masked_batch)
            logits_for_labels = np.asarray([y[i] for y, i in zip(logits, y_batch)])
            scores[i] = logits_for_labels

        if self.return_auc_per_sample:
            return calculate_auc(np.asarray(scores))

        return scores

    @property
    def auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean([calculate_auc(np.array(curve)) for curve in self.last_results]) # pragma: not covered
