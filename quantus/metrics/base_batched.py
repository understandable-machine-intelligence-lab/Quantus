"""This module implements the base class for creating evaluation metrics."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import math
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from .base import Metric
from ..helpers import asserts


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
        softmax: bool,
        default_plot_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
        **kwargs,
    ):
        """
        Initialise the Metric base class.

        Each of the defined metrics in Quantus, inherits from Metric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply evaluate_instance() on each
                      instance and finally call custom_preprocess().
                      To use this method the child Metric needs to implement
                      evaluate_instance().
        - general_preprocess(): Prepares all necessary data structures for evaluation.
                                Will call custom_preprocess() at the end.

        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed.

        """

        # Initialize super-class with passed parameters
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            softmax=softmax,
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
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func (callable): Callable generating attributions.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method.
        batch_size (int): batch size for evaluation, default = 64.

        Returns
        -------
        last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
        --------
        # Enable GPU.
        >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
        >> model = LeNet()
        >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

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
        # TODO: Move this type of warning into the for loop.
        #       We want to be flexible and add custom metric call kwargs to
        #       the calling of self.evaluate_instance().
        #       Check there if the implemented method of the child class
        #       has this keyword specified. If not, raise a warning.
        #warn_func.deprecation_warnings(kwargs)
        #asserts.check_kwargs(kwargs)

        (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        ) = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )

        # create generator for generating batches
        batch_generator = self.generate_batches(
            x_batch, y_batch, a_batch, s_batch,
            batch_size=batch_size,
            display_progressbar=self.display_progressbar,
        )

        n_batches = self.get_number_of_batches(arr=x_batch, batch_size=batch_size)
        # TODO: initialize correct length of last results
        self.last_results = []
        # We use a tailing underscore to prevent confusion with the passed parameters.
        # TODO: rename kwargs of __call__() method accordingly or else this still will be confusing.
        for x_batch_, y_batch_, a_batch_, s_batch_ in batch_generator:
            result = self.process_batch(
                model=model,
                x_batch=x_batch_,
                y_batch=y_batch_,
                a_batch=a_batch_,
                s_batch=s_batch_,
                **kwargs,
            )
            # TODO: put in correct idx instead of extending
            self.last_results.extend(result)

        # Call post-processing
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        self.all_results.append(self.last_results)
        return self.last_results

    @abstractmethod
    def process_batch(
            self,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            s_batch: Optional[np.ndarray] = None,
            **kwargs,
    ):
        raise NotImplementedError()

    @staticmethod
    def get_number_of_batches(arr: Sequence, batch_size: int):
        return math.ceil(len(arr)/batch_size)

    def generate_batches(
            self,
            *iterables: np.ndarray, batch_size: int,
            display_progressbar: bool = False,
    ):
        if iterables[0] is None:
            raise ValueError("first iterable must not be None!")

        iterables = list(iterables)
        n_instances = len(iterables[0])
        n_batches = self.get_number_of_batches(iterables[0], batch_size=batch_size)

        # check if any of the iterables is None and replace with list of None
        for i in range(len(iterables)):
            if iterables[i] is None:
                iterables[i] = [None for _ in range(n_instances)]

        if not all(len(iterable) == len(iterables[0]) for iterable in iterables):
            raise ValueError("number of instances needs to be equal for all iterables")

        iterator = tqdm(
            range(0, n_batches),
            total=n_batches,
            disable=not display_progressbar,
        )

        for batch_idx in iterator:
            batch_start = batch_size * batch_idx
            batch_end = min(batch_size * (batch_idx + 1), n_instances)
            batch = tuple(iterable[batch_start:batch_end] for iterable in iterables)
            yield batch


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
        n_steps: int,
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

        self.n_steps = n_steps
        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        self.perturb_func_kwargs = perturb_func_kwargs

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
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            n_steps=self.n_steps,
            **kwargs,
        )
