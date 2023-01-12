from __future__ import annotations

import functools
from abc import abstractmethod
import numpy as np
from typing import List, Optional, Dict, Callable
from quantus.metrics.base_batched import BatchedPerturbationMetric

from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
    PlainTextPerturbFn,
    PersistFn,
    NormaliseFn,
    NumericalPerturbFn,
    PerturbationType,
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


class BatchedTextClassificationMetric(BatchedPerturbationMetric):
    def __init__(
        self,
        abs: bool,  # noqa
        normalise: bool,
        normalise_func: Optional[NormaliseFn],
        normalise_func_kwargs: Optional[Dict],
        perturbation_type: PerturbationType,
        perturb_func: PlainTextPerturbFn | NumericalPerturbFn,
        perturb_func_kwargs: Optional[Dict],
        return_aggregate: bool,
        aggregate_func: Optional[Callable],
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        persist_func: Optional[PersistFn],
        **kwargs,
    ):
        """
        Initialise the BatchedTextClassificationMetric base class.

        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call.
        perturb_func: callable
            Input perturbation function.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call..
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed.
        kwargs: optional
            Keyword arguments.
        perturbation_type:
            PerturbationType.plainText means perturb_func will be applied to plain text inputs.
            PerturbationType.latent_space means the perturb_func will be applied to sum of word and positional embeddings.
        persist_func: callable
            persist_func: Optional[Callable[[List[np.ndarray | float]], None]].
            If provided, this function will be called after each batch of data was evaluated.
            This can be used to save intermediate results, default=None.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            **kwargs,
        )

        if not isinstance(perturbation_type, PerturbationType):
            raise ValueError("Only enum values of type PerturbationType are allowed")

        self.noise_type = perturbation_type
        self.persist_func = persist_func

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

        """
        Parameters
        ----------
        model: `quantus.nlp.TextClassifier`
            Model that is subject to explanation.
        x_batch: List[str]
            A batch of plain text inputs.
        explain_func: Callable[[TextClassifier, List[str], np.ndarray, ...], List[Explanation]]
            Function used to generate explanations, default=`quantus.nlp.explain`.
            Must have signature Callable[[TextClassifier, List[str], np.ndarray, ...], List[Explanation]] if noise_type="plain_text".
            Must have signature Callable[[TextClassifier, np.ndarray, np.ndarray, ...], np.ndarray] if noise_type="latent".
        explain_func_kwargs: Optional[Dict]
            Kwargs passed to explain_func.
        y_batch: Optional[np.ndarray]
            Batch of labels for x_batch, default=None. If not provided, will use prediction result.
        a_batch: Optional[List[Tuple[List[str], np.ndarray]]]
            A list of pre-computed explanations for x_batch and y_batch.
        batch_size: int
            Integer defining the limit of batch size used for computations, default=64.
        kwargs:
            Unused.

        Returns
        -------
        result: np.ndarray | float
            Returns float if return_aggregate=True, otherwise np.ndarray.

        """

        # Run deprecation warnings.
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
            if self.persist_func is not None:
                self.persist_func(result)
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
    ) -> Dict[str, ...]:

        """
        Prepares all necessary variables for evaluation.

            - Creates predictions if not provided.
            - Creates attributions if necessary.
            - Calls custom_preprocess().
            - Normalises attributions if desired.
            - Takes absolute of attributions if desired.

        Parameters
        ----------

        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        batch_size: int
            Batch size to use during evaluation.

        Returns
        -------
        tuple
            A general preprocess.

        """

        self.explain_func_kwargs = value_or_default(
            explain_func_kwargs, lambda: {}
        )  # noqa
        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func  # noqa
        y_batch = value_or_default(
            y_batch, lambda: model.predict(x_batch).argmax(axis=-1)
        )

        if a_batch is None:
            a_batch = self._generate_a_batch(model, x_batch, y_batch, batch_size)

        if self.normalise:
            a_batch = normalise_attributions(
                a_batch,
                functools.partial(self.normalise_func, **self.normalise_func_kwargs),
            )
        if self.abs:
            a_batch = abs_attributions(a_batch)

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

    def _generate_a_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        batch_size: int,
    ) -> List[Explanation]:

        explain_fn = functools.partial(self.explain_func, **self.explain_func_kwargs)

        if len(x_batch) <= batch_size:
            return explain_fn(model, x_batch, y_batch)

        batched_x = batch_list(x_batch, batch_size)
        batched_y = batch_list(y_batch.tolist(), batch_size)  # noqa

        a_batch = []
        for x, y in zip(batched_x, batched_y):
            a_batch.extend(explain_fn(model, x, np.asarray(y)))

        return a_batch

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
        raise NotImplementedError()
