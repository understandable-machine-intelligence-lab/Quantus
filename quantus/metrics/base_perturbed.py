"""This module implements the base class for creating evaluation metrics."""
# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    List
)
import warnings

import numpy as np

from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base import Metric


class PerturbationMetric(Metric, ABC):
    """
    Implementation base PertubationMetric class.

    Metric categories such as Faithfulness and Robustness share certain characteristics when it comes to perturbations.
    As follows, this metric class is created which has additional attributes for perturbations.

    Attributes:
        - name: The name of the metric.
        - data_applicability: The data types that the metric implementation currently supports.
        - model_applicability: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Callable,
        normalise_func_kwargs: Optional[Dict[str, Any]],
        perturb_func: Callable,
        perturb_func_kwargs: Optional[Dict[str, Any]],
        return_aggregate: bool,
        aggregate_func: Callable,
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):
        """
        Initialise the PerturbationMetric base class.

        Parameters
        ----------
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
        perturb_func_kwargs: dict, optional
            Keyword arguments to be passed to perturb_func.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed.
        kwargs: optional
            Keyword arguments.
        """

        # Initialize super-class with passed parameters
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

        # TODO: do we really need separate 150+ lines long class just to reuse 4 lines of code?
        # Save perturbation metric attributes.
        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        self.perturb_func_kwargs = perturb_func_kwargs
    
    def changed_prediction_indices(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            x_perturbed: np.ndarray
    ) -> List[int]:
        
        """
        Find indices in batch, for which predicted label has changed after applying perturbation.
        If metric has no `return_nan_when_prediction_changes` attribute, or it is False, will return empty list.
        
        Parameters
        ----------
        model:
        x_batch:
            Batch of original inputs provided by user.
        x_perturbed:
            Batch of inputs after applying perturbation.

        Returns
        -------
        
        changed_idx:
            List of indices in batch, for which predicted label has changed afer.

        """
        
        if hasattr(self, "return_nan_when_prediction_changes"):
            attr_name = "return_nan_when_prediction_changes"
        elif hasattr(self, "_return_nan_when_prediction_changes"):
            attr_name = "_return_nan_when_prediction_changes"
        else:
            warnings.warn("Called changed_prediction_indices(), from a metric, "
                          "without `return_nan_when_prediction_changes` instance attribute, this is unexpected.")
            return []
            
        if not getattr(self, attr_name):
            return []
        
        labels_before = model.predict(x_batch).argmax(axis=-1)
        labels_after = model.predict(x_perturbed).argmax(axis=-1)
        changed_idx = np.reshape(np.argwhere(labels_before != labels_after), -1)
        return changed_idx.tolist()
