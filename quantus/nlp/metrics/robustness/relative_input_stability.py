# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from quantus.helpers.relative_stability import relative_input_stability_objective
from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.functions.perturb_func import spelling_replacement
from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
    NormaliseFn,
    PerturbFn,
)
from quantus.nlp.helpers.utils import get_input_ids, get_scores, safe_as_array
from quantus.nlp.metrics.robustness.internal.relative_stability import RelativeStability


class RelativeInputStability(RelativeStability):
    """
    Relative Input Stability leverages the stability of an explanation with respect to the change in the input data

    :math:`RIS(x, x', e_x, e_x') = max \\frac{||\\frac{e_x - e_{x'}}{e_x}||_p}{max (||\\frac{x - x'}{x}||_p, \epsilon_{min})}`

    References:
        1) Chirag Agarwal, et. al., 2022. "Rethinking stability for attribution based explanations.", https://arxiv.org/abs/2203.06877
    """

    def __init__(
        self,
        abs: bool = False,  # noqa
        normalise: bool = True,
        normalise_func: NormaliseFn = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        perturb_func: PerturbFn = spelling_replacement,
        perturb_func_kwargs: Optional[Dict] = None,
        nr_samples: int = 50,
    ):
        """

        Parameters
        ----------
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
        perturb_func: callable
            Input perturbation function. If None, the default value is used, default=spelling_replacement.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        nr_samples: int
            The number of samples iterated, default=50.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func_kwargs=normalise_func_kwargs,
            normalise_func=normalise_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            nr_samples=nr_samples,
        )

    def __call__(
        self,
        model: TextClassifier,
        x_batch: List[str],
        *,
        y_batch: Optional[np.ndarray] = None,
        a_batch: Optional[List[Explanation] | np.ndarray] = None,
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

    def _compute_objective_plain_text(
        self,
        model: TextClassifier,
        x_batch: List[str],
        x_batch_perturbed: List[str],
        a_batch: List[Explanation],
        a_batch_perturbed: List[Explanation],
    ) -> np.ndarray:
        e_x = get_scores(a_batch)
        e_xs = get_scores(a_batch_perturbed)
        x = safe_as_array(model.embedding_lookup(get_input_ids(x_batch, model)[0]))
        x_s = safe_as_array(
            model.embedding_lookup(get_input_ids(x_batch_perturbed, model)[0])
        )

        return relative_input_stability_objective(x, x_s, e_x, e_xs)

    def _compute_objective_latent_space(
        self,
        model: TextClassifier,
        x_batch: np.ndarray,
        x_batch_perturbed: np.ndarray,
        a_batch: np.ndarray,
        a_batch_perturbed: np.ndarray,
        predict_kwargs: Dict,
    ) -> np.ndarray:
        return relative_input_stability_objective(
            x_batch, x_batch_perturbed, a_batch, a_batch_perturbed
        )
