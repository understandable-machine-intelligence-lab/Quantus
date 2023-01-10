from __future__ import annotations

from abc import abstractmethod
import numpy as np
from typing import List, Optional, Dict, overload
from quantus.metrics.base_batched import BatchedPerturbationMetric

from quantus.nlp.helpers.types import ExplainFn, NormaliseFn, Explanation, PerturbFn
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import wrap_model, value_or_default
from quantus.helpers.warn import deprecation_warnings, check_kwargs


class BatchedTextClassificationMetric(BatchedPerturbationMetric):
    explain_func: ExplainFn
    explain_func_kwargs: Dict
    perturb_func: PerturbFn
    perturb_func_kwargs: Dict
    normalise: bool
    abs: bool
    normalise_func: NormaliseFn
    normalise_func_kwargs: Dict
    model_predict_kwargs: Dict
    all_result: List[float]
    last_result: List[float]

    def __call__(
        self,
        model,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        *,
        a_batch: Optional[Explanation],
        tokenizer,
        explain_func: ExplainFn,
        explain_func_kwargs: Optional[Dict],
        model_init_kwargs: Optional[Dict],
        model_predict_kwargs: Optional[Dict],
        batch_size: int = 64,
        **kwargs,
    ) -> np.ndarray | float:

        # Run deprecation warnings.
        check_kwargs(kwargs)
        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            model_init_kwargs=model_init_kwargs,
            tokenizer=tokenizer,
        )

        # Create generator for generating batches.
        batch_generator = self.generate_batches(
            data=data,
            batch_size=batch_size,
        )

        self.last_results = []
        for data_batch in batch_generator:
            result = self.evaluate_batch(**data_batch)
            self.last_results.extend(result)

        # Call post-processing.
        self.custom_postprocess(**data)

        # Append content of last results to all results.
        self.all_results.append(self.last_results)

        return np.asarray(self.last_results)

    def general_preprocess(
        self,
        *,
        model,
        tokenizer: Optional,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Optional[Dict],
        model_init_kwargs: Optional[Dict],
        model_predict_kwargs: Optional[Dict],
        **kwargs,
    ) -> Dict:

        self.explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})
        # self.normalise_func_kwargs = value_or_default()
        model_init_kwargs = value_or_default(model_init_kwargs, lambda: {})
        self.model_predict_kwargs = value_or_default(model_predict_kwargs, lambda: {})

        # Wrap the model into an interface.
        model = wrap_model(model, tokenizer, model_init_kwargs)

        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func

        a_batch = value_or_default(
            a_batch,
            lambda: self.explain_func(x_batch, y_batch, model, **explain_func_kwargs),
        )

        # Initialize data dictionary.
        data = {
            "model": model,
            "x_batch": x_batch,
            "y_batch": y_batch,
            "a_batch": a_batch,
            "s_batch": None,
            "custom_batch": None,
        }

        # Call custom pre-processing from inheriting class.
        custom_preprocess_dict = self.custom_preprocess(**data)

        # Save data coming from custom preprocess to data dict.
        if custom_preprocess_dict:
            for key, value in custom_preprocess_dict.items():
                data[key] = value

        # Normalise with specified keyword arguments if requested.
        if self.normalise:
            data["a_batch"] = self.normalise_func(
                data["a_batch"], **self.normalise_func_kwargs
            )

        # Take absolute if requested.
        if self.abs:
            data["a_batch"] = np.abs(data["a_batch"])

        return data

    @abstractmethod
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        **kwargs,
    ) -> np.ndarray | float:
        raise NotImplementedError()
