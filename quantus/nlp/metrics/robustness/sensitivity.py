from __future__ import annotations

import numpy as np

from typing import List, Callable, Optional, Dict
from abc import abstractmethod
from quantus.nlp.functions.perturbation_function import spelling_replacement
from quantus.helpers.asserts import attributes_check
from quantus.functions.norm_func import fro_norm
from quantus.nlp.helpers.types import (
    PerturbFn,
    SimilarityFn,
    NormaliseFn,
    Explanation,
    ExplainFn,
)
from quantus.nlp.metrics.text_classification_metric import (
    BatchedTextClassificationMetric,
)
from quantus.nlp.helpers.utils import value_or_default, to_numpy
from nlp.helpers.types import TextClassifier
from quantus.nlp.helpers.warn import warn_perturbation_caused_no_change
from quantus.nlp.functions.similarity_func import difference


class Sensitivity(BatchedTextClassificationMetric):

    similarity_func: SimilarityFn
    similarity_func_kwargs: Dict

    @attributes_check
    def __init__(
        self,
        *,
        similarity_func: Optional[SimilarityFn] = None,
        similarity_func_kwargs: Optional[Dict] = None,
        norm_numerator: Optional[Callable] = None,
        norm_denominator: Optional[Callable] = None,
        nr_samples: int = 200,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = None,
        normalise_func_kwargs: Optional[Dict] = None,
        perturb_func: PerturbFn = None,
        perturb_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_nan_when_prediction_changes: bool = False,
        **kwargs,
    ):

        perturb_func = value_or_default(perturb_func, lambda: spelling_replacement)
        perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {"k": 1})

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.nr_samples = nr_samples
        self.similarity_func = value_or_default(similarity_func, lambda: difference)
        self.similarity_func_kwargs = value_or_default(
            similarity_func_kwargs, lambda: {"padded": True}
        )

        self.norm_numerator = value_or_default(norm_numerator, lambda: fro_norm)
        self.norm_denominator = value_or_default(norm_denominator, lambda: fro_norm)
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes

    def __call__(
        self,
        model,
        x_batch: List[str],
        y_batch: np.ndarray,
        *,
        explain_func: ExplainFn,
        model_init_kwargs: Optional[Dict] = None,
        tokenizer: Optional = None,
        explain_func_kwargs: Optional[Dict] = None,
        a_batch: Optional[np.ndarray] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray:

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            model_init_kwargs=model_init_kwargs,
            tokenizer=tokenizer,
            **kwargs,
        )

    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:

        batch_size = len(x_batch)
        similarities = np.zeros((batch_size, self.nr_samples)) * np.nan
        input_ids = model.tokenizer.tokenize(x_batch)
        if isinstance(input_ids, Dict):
            input_ids = input_ids["input_ids"]
        x_batch_embeddings = model.embedding_lookup(input_ids)
        x_batch_embeddings = to_numpy(x_batch_embeddings)

        for step_id in range(self.nr_samples):

            # Perturb input.
            x_perturbed = self.perturb_func(x_batch, **self.perturb_func_kwargs)
            warn_perturbation_caused_no_change(x_batch, x_perturbed)

            changed_prediction_indices = self._indexes_of_changed_predictions(
                model, x_batch, x_perturbed
            )
            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_func(
                x_perturbed, y_batch, model, **self.explain_func_kwargs
            )

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed,
                    **self.normalise_func_kwargs,
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            # Measure similarity for each instance separately.
            for instance_id in range(batch_size):

                if self._prediction_changed(changed_prediction_indices, instance_id):
                    similarities[instance_id, step_id] = np.nan
                    continue

                sensitivities = self.similarity_func(
                    a_batch[instance_id],
                    a_perturbed[instance_id],
                    **self.similarity_func_kwargs,
                )

                numerator = self.norm_numerator(sensitivities)
                denominator = self.norm_denominator(
                    x_batch_embeddings[instance_id].flatten()
                )

                sensitivities_norm = numerator / denominator
                similarities[instance_id, step_id] = sensitivities_norm
        return self.aggregate_steps(similarities)

    def _indexes_of_changed_predictions(
        self, model: TextClassifier, x_batch: List[str], x_batch_perturbed: List[str]
    ) -> np.ndarray | List:
        # TODO move out in separate small class
        if not self.return_nan_when_prediction_changes:
            return []

        labels_before = model.predict(x_batch).argmax(axis=-1)
        labels_after = model.predict(x_batch_perturbed).argmax(axis=-1)
        return np.argwhere(labels_before != labels_after).reshape(-1)

    def _prediction_changed(
        self, changed_prediction_indices: np.ndarray | List, index: int
    ) -> bool:
        if not self.return_nan_when_prediction_changes:
            return False
        return index in changed_prediction_indices

    @abstractmethod
    def aggregate_steps(self, arr: np.ndarray) -> float | np.ndarray:
        pass
