"""This module implements the base class for creating evaluation metrics."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import inspect
import math
import re
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from quantus.metrics.base import Metric
from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface


class BatchedMetric(Metric):
    """
    Implementation base BatchedMetric class.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        return_aggregate: bool,
        aggregate_func: Optional[Callable],
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):
        """
        Initialise the BatchedMetric base class.

        Each of the defined metrics in Quantus, inherits from Metric or BatchedMetric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply evaluate_instance() on each
                      instance and finally call custom_preprocess().
                      To use this method the child BatchedMetric needs to implement
                      evaluate_instance().
        - general_preprocess(): Prepares all necessary data structures for evaluation.
                                Will call custom_preprocess() at the end.

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

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        The content of last_results will be appended to all_results (list) at the end of
        the evaluation call.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
                This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        custom_batch: any
            Any object that can be passed to the evaluation process.
            Gives flexibility to the user to adapt for implementing their own metric.
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        last_results: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
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

        return self.last_results

    @abstractmethod
    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray,
    ):
        """
        Evaluates model and attributes on a single data batch and returns the batched evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        s_batch: np.ndarray
            The segmentation to be evaluated on a batch-basis.

        Returns
        -------
        np.ndarray
            The batched evaluation results.
        """
        raise NotImplementedError()

    @staticmethod
    def get_number_of_batches(n_instances: int, batch_size: int) -> int:
        """
        Get the number of batches given number of samples/ instances and a batch size.

        Parameters
        ----------
        n_instances: int
            The number of instances.
        batch_size: int
            The batch size.

        Returns
        -------
        integer
        """
        return math.ceil(n_instances / batch_size)

    def generate_batches(
        self,
        data: Dict[str, Any],
        batch_size: int,
    ):
        """
        Creates iterator to iterate over all batched instances in data dictionary.
        Each iterator output element is a keyword argument dictionary with
        string keys.

        Each item key in the input data dictionary has to be of type string.
        - If the item value is not a sequence, the respective item key/value pair
          will be written to each iterator output dictionary.
        - If the item value is a sequence and the item key ends with '_batch',
          a check will be made to make sure length matches number of instances.
          The values of the batch instances in the sequence will be added to the respective
          iterator output dictionary with the '_batch' suffix removed.
        - If the item value is a sequence but doesn't end with '_batch', it will be treated
          as a simple value and the respective item key/value pair will be
          written to each iterator output dictionary.

        Parameters
        ----------
        data: dict[str, any]
            The data input dictionary.
        batch_size: int
            The batch size to be used.

        Returns
        -------
        iterator
            Each iterator output element is a keyword argument dictionary (string keys).

        """
        n_instances = len(data["x_batch"])

        single_value_kwargs: Dict[str, Any] = {}
        batched_value_kwargs: Dict[str, Any] = {}

        for key, value in list(data.items()):
            # If data-value is not a Sequence or a string, create list of value with length of n_instances.
            if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, str):
                single_value_kwargs[key] = value

            # If data-value is a sequence and ends with '_batch', only check for correct length.
            elif key.endswith("_batch"):
                if len(value) != n_instances:
                    # Sequence has to have correct length.
                    raise ValueError(
                        f"'{key}' has incorrect length (expected: {n_instances}, is: {len(value)})"
                    )
                else:
                    batched_value_kwargs[key] = value

            # If data-value is a sequence and doesn't end with '_batch', create
            # list of repeated sequences with length of n_instances.
            else:
                single_value_kwargs[key] = [value for _ in range(n_instances)]

        n_batches = self.get_number_of_batches(
            n_instances=n_instances, batch_size=batch_size
        )

        # Create iterator for batch index.
        iterator = tqdm(
            range(0, n_batches),
            total=n_batches,
            disable=not self.display_progressbar,
        )

        # Iterate over batch index
        for batch_idx in iterator:
            # Calculate instance index for start and end of batch.
            batch_start = batch_size * batch_idx
            batch_end = min(batch_size * (batch_idx + 1), n_instances)

            # Create batch dictionary with all specified batch instance values
            batch = {
                key: value[batch_start:batch_end]
                for key, value in batched_value_kwargs.items()
            }

            # Yield batch dictionary including single value keyword arguments.
            yield {**batch, **single_value_kwargs}

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
        **kwargs,
    ) -> Any:
        """
        This method from the parent Metric class needs to be defined to implement this abstract class.
        However we use evalaute_batch() instead for BatchedMetric.

        Returns
        -------
        Any
        """
        raise NotImplementedError(
            "evaluate_instance() not implemented for BatchedMetric"
        )


class BatchedPerturbationMetric(BatchedMetric):
    """
    Implementation base BatchedPertubationMetric class.

    This batched metric has additional attributes for perturbations.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict[str, Any]],
        perturb_func: Callable,
        perturb_func_kwargs: Optional[Dict[str, Any]],
        return_aggregate: bool,
        aggregate_func: Optional[Callable],
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
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

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        self.perturb_func_kwargs = perturb_func_kwargs

    @abstractmethod
    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates model and attributes on a single data batch and returns the batched evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        s_batch: np.ndarray
            The segmentation to be evaluated on a batch-basis.

        Returns
        -------
        np.ndarray
            The batched evaluation results.
        """
        raise NotImplementedError()

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
        **kwargs,
    ) -> Any:
        """
        This method from the parent Metric class needs to be defined to implement this abstract class.
        However we use evalaute_batch() instead for BatchedMetric.

        Parameters
        ----------
        kwargs: optional
            Keyword arguments.
        """
        raise NotImplementedError(
            "evaluate_instance() not implemented for BatchedPerturbationMetric"
        )
