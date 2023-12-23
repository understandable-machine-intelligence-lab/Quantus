# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import abc
import logging

from typing import List, Tuple, Dict
import functools
from quantus.metrics.base import Metric
from quantus.helpers.collection_utils import safe_as_array, value_or_default
from quantus.functions.perturb_func import perturb_batch as perturb_batch_fn
from quantus.helpers.nlp_utils import is_plain_text_perturbation
from quantus.helpers.typing_utils import TextClassifier, ModelInterface, Explanation
from abc import abstractmethod
import numpy as np

"""Aliases to smoothen transition to uniform metric API."""


class BatchedMetric(Metric, abc.ABC):

    """Alias to quantus.Metric, will be removed in next major release."""

    def __new__(cls, *args, **kwargs):
        logging.warning(
            "BatchedMetric was deprecated, since it is just an alias to Metric."
            " Please subclass Metric directly."
        )
        super().__new__(*args, **kwargs)



class BatchedPerturbationMetric(Metric):
    """
    Implementation base BatchedPertubationMetric class.

    This batched metric has additional attributes for perturbations.
    """

    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func,
        normalise_func_kwargs,
        perturb_func,
        perturb_func_kwargs,
        return_aggregate: bool,
        aggregate_func,
        default_plot_func,
        disable_warnings: bool,
        display_progressbar: bool,
        nr_samples: int = None,
        return_nan_when_prediction_changes: bool = None,
        **kwargs,
    ):
        """
        Initialise the PerturbationMetric base class.

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
        """

        # Initialise super-class with passed parameters.
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
            **kwargs,
        )

        # Save perturbation metric attributes.
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes
        self.nr_samples = nr_samples

    def changed_prediction_indices(
        self,
        model,
        x_batch,
        x_perturbed,
        **kwargs,
    ):
        """Predict on x_batch and x_perturbed, return indices of mismatched labels."""
        if not self.return_nan_when_prediction_changes:
            return []
        og_labels = np.argmax(safe_as_array(model.predict(x_batch, **kwargs)), axis=-1)
        perturbed_labels = np.argmax(
            safe_as_array(model.predict(x_perturbed, **kwargs)), axis=-1
        )
        return np.reshape(np.argwhere(og_labels != perturbed_labels), -1)

    def perturb_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """Apply self.perturb_fn to batch of images."""
        batch_size = x_batch.shape[0]
        size = np.size(x_batch[0])
        ndim = np.ndim(x_batch[0])

        return perturb_batch_fn(
            perturb_func=self.perturb_func,
            indices=np.tile(np.arange(0, size), (batch_size, 1)),
            indexed_axes=np.arange(0, ndim),
            arr=x_batch,
            **self.perturb_func_kwargs,
        )

    def batch_preprocess(
        self,
        model: ModelInterface | TextClassifier,
        x_batch: np.ndarray | List[str],
        y_batch: np.ndarray | None,
        a_batch: np.ndarray | List[Explanation] | None,
        **kwargs
    ):
        """
        For text classification + plain-text perturb_func we need to pre-compute
        perturbations, and then pad all them, so all plain-text sequences have same number of tokens.
        We also hook into tokenizer.batch_encode to prevent removing of added padded tokens.
        """
        if "NLP" not in self.data_domain_applicability:
            return super().batch_preprocess(model, x_batch, y_batch, a_batch)
        if not is_plain_text_perturbation(self.perturb_func):
            return super().batch_preprocess(model, x_batch, y_batch, a_batch)

        batch_size = len(x_batch)
        # For plain text we need to first collect perturbations, then
        # "pre-tokenize" them, so we end up with sequences all the same length.
        x_perturbed_batches = [x_batch] + [
            self.perturb_func(x_batch, **self.perturb_func_kwargs)
            for _ in range(self.nr_samples)
        ]
        x_perturbed_batches = np.reshape(x_perturbed_batches, -1).tolist()
        x_perturbed_ids, _ = model.tokenizer.get_input_ids(x_perturbed_batches)
        x_perturbed_batches = model.tokenizer.batch_decode(x_perturbed_ids)
        x_batch, x_perturbed_batches = (
            x_perturbed_batches[:batch_size],
            x_perturbed_batches[batch_size:],
        )
        x_perturbed_batches = np.reshape(x_perturbed_batches, (self.nr_samples, -1))
        x_perturbed_batches = [i.tolist() for i in x_perturbed_batches]

        # Leave padding tokens in.
        model.tokenizer.batch_encode = functools.partial(
            model.tokenizer.batch_encode, add_special_tokens=False
        )
        x_batch, y_batch, a_batch, _ = super().batch_preprocess(
            model, x_batch, y_batch, None
        )
        return x_batch, y_batch, a_batch, x_perturbed_batches

    def batch_postprocess(
        self,
        model: ModelInterface | TextClassifier,
        x_batch: np.ndarray | List[str],
        y_batch: np.ndarray,
        a_batch: np.ndarray | List[Explanation],
        s_batch: np.ndarray | None,
        score: np.ndarray,
    ) -> np.ndarray:
        """
        Since for text classification + plain-text perturb_func we modified tokenizer.batch_encode's behaviour,
        here we restore the default one.
        """
        if isinstance(x_batch[0], str):
            if is_plain_text_perturbation(self.perturb_func):
                model.tokenizer.batch_encode = model.tokenizer.batch_encode.func
        return super().batch_postprocess(
            model, x_batch, y_batch, a_batch, s_batch, score
        )

    def evaluate_batch(
        self,
        model: ModelInterface | TextClassifier,
        x_batch: np.ndarray | List[str],
        y_batch: np.ndarray,
        a_batch,
        s_batch: np.ndarray = None,
        custom_batch=None,
    ) -> np.ndarray:
        similarities = []

        for step_id in range(self.nr_samples):
            if isinstance(model, TextClassifier):
                if is_plain_text_perturbation(self.perturb_func):
                    x_perturbed = custom_batch[step_id]
                    # Generate explanation based on perturbed input x.
                    a_perturbed = self.explain_batch(model, x_perturbed, y_batch)
                    x_batch_embeddings, _ = model.get_embeddings(x_batch)
                    predict_kwargs = {}
                else:
                    x_batch_embeddings, predict_kwargs = model.get_embeddings(x_batch)
                    x_perturbed = self.perturb_batch(x_batch_embeddings)
                    # Generate explanation based on perturbed input x.
                    a_perturbed = self.explain_batch(
                        model, x_perturbed, y_batch, **predict_kwargs
                    )
            else:
                x_perturbed = self.perturb_batch(x_batch)
                a_perturbed = self.explain_batch(model, x_perturbed, y_batch)
                predict_kwargs = {}

            similarities.append(
                self.evaluate_sample(
                    model,
                    x_batch,
                    x_perturbed,
                    a_batch,
                    a_perturbed,
                    y_batch,
                    predict_kwargs,
                )
            )
        return self.reduce_samples(similarities)

    @abstractmethod
    def evaluate_sample(
        self, model, x_batch, x_perturbed, a_batch, a_perturbed, y_batch, predict_kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def reduce_samples(self, scores):
        raise NotImplementedError

