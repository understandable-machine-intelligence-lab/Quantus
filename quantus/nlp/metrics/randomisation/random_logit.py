# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.


from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from quantus.functions.similarity_func import correlation_kendall_tau
from quantus.helpers.utils import off_label_choice
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation, NormaliseFn, SimilarityFn
from quantus.nlp.helpers.utils import get_scores
from quantus.nlp.metrics.text_classification_metric import TextClassificationMetric


class RandomLogit(TextClassificationMetric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.

    References:
        1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
        Attributions Fail." ICML (2020): 9046-9057.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        similarity_func: SimilarityFn = correlation_kendall_tau,
    ):
        """

        Parameters
        ----------
        num_classes: integer
            Number of prediction classes in the input, default=2.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        similarity_func: callable
            Similarity function applied to compare input and perturbed input, default=`quantus.correlation_kendal_tau`.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
        )
        self.num_classes = num_classes
        self.similarity_func = similarity_func

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
    ) -> np.ndarray:
        """

        Parameters
        ----------
        model:
            Torch or tensorflow model that is subject to explanation. Most probably, you will want to use
            `quantus.nlp.TorchHuggingFaceTextClassifier` or `quantus.nlp.TensorFlowHuggingFaceTextClassifier`,
            for out-of-the box support for models from Huggingface hub.
        x_batch:
            list, which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.
        a_batch:
            Pre-computed attributions i.e., explanations. Token and scores as well as scores only are supported.
        explain_func:
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        batch_size:
            Indicates size of batches, in which input dataset will be splitted.

        Returns
        -------

        score:
            np.ndarray of scores.

        """
        return super().__call__(
            model,
            x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            batch_size=batch_size,
        )

    def _evaluate_batch(  # type: ignore
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        **kwargs,
    ) -> np.ndarray:
        y_off = off_label_choice(y_batch, self.num_classes)

        # Explain against a random class.
        a_perturbed = self._explain_batch(
            model, x_batch, y_off, explain_func, explain_func_kwargs
        )
        return self.similarity_func(
            get_scores(a_batch),
            get_scores(a_perturbed),
        )
