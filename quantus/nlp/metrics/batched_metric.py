from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import List, Optional, Dict
from functools import partial
from quantus.metrics.base_batched import BatchedMetric as Base
from functools import partial

from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
)
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import (
    value_or_default,
    batch_list,
    normalise_attributions,
    abs_attributions,
)
from quantus.helpers.warn import check_kwargs
from quantus.nlp.functions.explanation_func import explain


class BatchedMetric(Base):
    def __call__(
        self,
        model: TextClassifier,
        x_batch: List[str],
        *,
        y_batch: Optional[np.ndarray] = None,
        a_batch: Optional[List[Explanation]] = None,
        explain_func: ExplainFn = explain,
        explain_func_kwargs: Optional[Dict] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[np.ndarray | float]:
        check_kwargs(kwargs)
        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
        )

        # Create generator for generating batches.
        batch_generator = self.generate_batches(
            data=data,  # noqa
            batch_size=batch_size,
        )

        self.last_results = []
        for data_batch in batch_generator:
            result = self.evaluate_batch(**data_batch)
            self.last_results.extend(result)

        # Call post-processing.
        self.custom_postprocess(**data)  # noqa

        # Append content of last results to all results.
        self.all_results.append(self.last_results)

        return self.last_results

    def general_preprocess(
        self,
        *,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Optional[Dict],
        batch_size: int,
        **kwargs,
    ) -> Dict:
        self.explain_func_kwargs = value_or_default(
            explain_func_kwargs, lambda: {}
        )  # noqa
        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func  # noqa
        y_batch = value_or_default(
            y_batch, lambda: model.predict(x_batch).argmax(axis=-1)
        )

        a_batch = value_or_default(
            a_batch,
            lambda: self._generate_a_batches(model, x_batch, y_batch, batch_size),
        )
        a_batch = self.normalise_a_batch(a_batch)

        # Initialize data dictionary.
        data = {
            "model": model,
            "x_batch": x_batch,
            "y_batch": y_batch,
            "a_batch": a_batch,
            # For compatibility reasons we need to provide "s_batch" and "custom_batch" keys.
            "s_batch": None,
            "custom_batch": None,
        }

        # Call custom pre-processing from inheriting class.
        custom_preprocess_dict = self.custom_preprocess(**data)

        # Save data coming from custom preprocess to data dict.
        if custom_preprocess_dict:
            for key, value in custom_preprocess_dict.items():
                data[key] = value

        return data

    def normalise_a_batch(
        self, a_batch: List[Explanation] | np.ndarray
    ) -> List[Explanation] | np.ndarray:
        if self.normalise:
            normalise_fn = partial(self.normalise_func, **self.normalise_func_kwargs)
            if not isinstance(a_batch, np.ndarray):
                normalise_fn = partial(
                    normalise_attributions, normalise_fn=normalise_fn
                )
            a_batch = normalise_fn(a_batch)

        if self.abs:
            if isinstance(a_batch, np.ndarray):
                abs_func = np.abs
            else:
                abs_func = abs_attributions
            a_batch = abs_func(a_batch)

        return a_batch

    def _generate_a_batches(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        batch_size: Optional[int],
    ) -> List[Explanation] | np.ndarray:
        explain_fn = partial(self.explain_func, **self.explain_func_kwargs)
        if len(x_batch) <= batch_size:
            return explain_fn(model, x_batch, y_batch)

        batched_x = batch_list(x_batch, batch_size)
        batched_y = batch_list(y_batch.tolist(), batch_size)  # noqa

        a_batches = []
        for x, y in zip(batched_x, batched_y):
            a_batches.extend(explain_fn(model, x, y))

        if isinstance(a_batches[0], np.ndarray):
            return np.asarray(a_batches)
        else:
            return a_batches

    @abstractmethod
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:
        """Must be implemented by respective metric class."""
        raise NotImplementedError
