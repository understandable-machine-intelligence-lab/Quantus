"""This module implements the base class for creating evaluation metrics."""
# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import inspect
import re
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Dict, Sequence, Optional, Tuple, Union, Collection
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface


class Metric:
    """
    Implementation of the base Metric class.
    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Callable,
        normalise_func_kwargs: Optional[Dict[str, Any]],
        return_aggregate: bool,
        aggregate_func: Callable,
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

        The content of last_results will be appended to all_results (list) at the end of
        the evaluation call.

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
        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        self.abs = abs
        self.normalise = normalise
        self.return_aggregate = return_aggregate
        self.aggregate_func = aggregate_func
        self.normalise_func = normalise_func

        if normalise_func_kwargs is None:
            normalise_func_kwargs = {}
        self.normalise_func_kwargs = normalise_func_kwargs

        self.default_plot_func = default_plot_func
        self.disable_warnings = disable_warnings
        self.display_progressbar = display_progressbar

        self.a_axes: Sequence[int] = None

        self.last_results: Any = []
        self.all_results: Any = []

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> Union[int, float, list, dict, Collection[Any], None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
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
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
            custom_batch=custom_batch,
        )

        self.last_results = [None for _ in x_batch]

        # Evaluate with instace given the metric.
        iterator = self.get_instance_iterator(data=data)
        for id_instance, data_instance in iterator:
            result = self.evaluate_instance(**data_instance)
            self.last_results[id_instance] = result

        # Call custom post-processing.
        self.custom_postprocess(**data)

        if self.return_aggregate:
            if self.aggregate_func:
                try:
                    self.last_results = [self.aggregate_func(self.last_results)]
                except:
                    print(
                        "The aggregation of evaluation scores failed. Check that "
                        "'aggregate_func' supplied is appropriate for the data "
                        "in 'last_results'."
                    )
            else:
                raise KeyError(
                    "Specify an 'aggregate_func' (Callable) to aggregate evaluation scores."
                )

        self.all_results.append(self.last_results)

        return self.last_results

    @abstractmethod
    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
    ) -> Any:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        Any
        """
        raise NotImplementedError()

    def general_preprocess(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Callable,
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: bool,
        device: Optional[str],
        custom_batch: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Prepares all necessary variables for evaluation.

            - Reshapes data to channel first layout.
            - Wraps model into ModelInterface.
            - Creates attributions if necessary.
            - Expands attributions to data shape (adds channel dimension).
            - Calls custom_preprocess().
            - Normalises attributions if desired.
            - Takes absolute of attributions if desired.
            - If no segmentation s_batch given, creates list of Nones with as many
              elements as there are data instances.
            - If no custom_batch given, creates list of Nones with as many
              elements as there are data instances.

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
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        tuple
            A general preprocess.

        """

        # Reshape input batch to channel first order:
        if not isinstance(channel_first, bool):  # None is not a boolean instance.
            channel_first = utils.infer_channel_first(x_batch)
        x_batch = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:

            # Use attribute value if not passed explicitly.
            model = utils.get_wrapped_model(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func
        if explain_func_kwargs is None:
            explain_func_kwargs = {}
        self.explain_func_kwargs = explain_func_kwargs

        # Include device in explain_func_kwargs.
        if device is not None and "device" not in self.explain_func_kwargs:
            self.explain_func_kwargs["device"] = device

        if a_batch is None:

            # Asserts.
            asserts.assert_explain_func(explain_func=self.explain_func)

            # Generate explanations.
            a_batch = self.explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.explain_func_kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

        # Infer attribution axes for perturbation function.
        self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

        # Initialize data dictionary.
        data = {
            "model": model,
            "x_batch": x_batch,
            "y_batch": y_batch,
            "a_batch": a_batch,
            "s_batch": s_batch,
            "custom_batch": custom_batch,
        }

        # Call custom pre-processing from inheriting class.
        custom_preprocess_dict = self.custom_preprocess(**data)

        # Save data coming from custom preprocess to data dict.
        if custom_preprocess_dict:
            for key, value in custom_preprocess_dict.items():
                data[key] = value

        # Remove custom_batch if not used.
        if data["custom_batch"] is None:
            del data["custom_batch"]

        # Normalise with specified keyword arguments if requested.
        if self.normalise:
            data["a_batch"] = self.normalise_func(
                a=data["a_batch"],
                normalise_axes=list(range(np.ndim(data["a_batch"])))[1:],
                **self.normalise_func_kwargs,
            )

        # Take absolute if requested.
        if self.abs:
            data["a_batch"] = np.abs(data["a_batch"])

        return data

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """
        Implement this method if you need custom preprocessing of data,
        model alteration or simply for creating/initialising additional
        attributes or assertions.

        If this method returns a dictionary, the keys (string) will be used as
        additional arguments for evaluate_instance().
        If the key ends with `_batch`, this suffix will be removed from the
        respective argument name when passed to evaluate_instance().
        If they key corresponds to the arguments `x_batch, y_batch, a_batch, s_batch`,
        these will be overwritten for passing `x, y, a, s` to `evaluate_instance()`.
        If this method returns None, no additional keyword arguments will be
        passed to `evaluate_instance()`.

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
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility to the inheriting metric to use for evaluation, can hold any variable.

        Returns
        -------
        dict, optional
            A dictionary which holds (optionally additional) preprocessed data to
           be included when calling `evaluate_instance()`.


        Examples
        --------
            # Custom Metric definition with additional keyword argument used in evaluate_instance():
            >>> def custom_preprocess(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x_batch: np.ndarray,
            >>>     y_batch: Optional[np.ndarray],
            >>>     a_batch: Optional[np.ndarray],
            >>>     s_batch: np.ndarray,
            >>>     custom_batch: Optional[np.ndarray],
            >>> ) -> Dict[str, Any]:
            >>>     return {'my_new_variable': np.mean(x_batch)}
            >>>
            >>> def evaluate_instance(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x: np.ndarray,
            >>>     y: Optional[np.ndarray],
            >>>     a: Optional[np.ndarray],
            >>>     s: np.ndarray,
            >>>     my_new_variable: np.float,
            >>> ) -> float:

            # Custom Metric definition with additional keyword argument that ends with `_batch`
            >>> def custom_preprocess(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x_batch: np.ndarray,
            >>>     y_batch: Optional[np.ndarray],
            >>>     a_batch: Optional[np.ndarray],
            >>>     s_batch: np.ndarray,
            >>>     custom_batch: Optional[np.ndarray],
            >>> ) -> Dict[str, Any]:
            >>>     return {'my_new_variable_batch': np.arange(len(x_batch))}
            >>>
            >>> def evaluate_instance(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x: np.ndarray,
            >>>     y: Optional[np.ndarray],
            >>>     a: Optional[np.ndarray],
            >>>     s: np.ndarray,
            >>>     my_new_variable: np.int,
            >>> ) -> float:

            # Custom Metric definition with transformation of an existing
            # keyword argument from `evaluate_instance()`
            >>> def custom_preprocess(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x_batch: np.ndarray,
            >>>     y_batch: Optional[np.ndarray],
            >>>     a_batch: Optional[np.ndarray],
            >>>     s_batch: np.ndarray,
            >>>     custom_batch: Optional[np.ndarray],
            >>> ) -> Dict[str, Any]:
            >>>     return {'x_batch': x_batch - np.mean(x_batch, axis=0)}
            >>>
            >>> def evaluate_instance(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x: np.ndarray,
            >>>     y: Optional[np.ndarray],
            >>>     a: Optional[np.ndarray],
            >>>     s: np.ndarray,
            >>> ) -> float:

            # Custom Metric definition with None returned in custom_preprocess(),
            # but with inplace-preprocessing and additional assertion.
            >>> def custom_preprocess(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x_batch: np.ndarray,
            >>>     y_batch: Optional[np.ndarray],
            >>>     a_batch: Optional[np.ndarray],
            >>>     s_batch: np.ndarray,
            >>>     custom_batch: Optional[np.ndarray],
            >>> ) -> None:
            >>>     if np.any(np.all(a_batch < 0, axis=0)):
            >>>         raise ValueError("Attributions must not be all negative")
            >>>
            >>>     x_batch -= np.mean(x_batch, axis=0)
            >>>
            >>>     return None
            >>>
            >>> def evaluate_instance(
            >>>     self,
            >>>     model: ModelInterface,
            >>>     x: np.ndarray,
            >>>     y: Optional[np.ndarray],
            >>>     a: Optional[np.ndarray],
            >>>     s: np.ndarray,
            >>> ) -> float:

        """
        pass

    def get_instance_iterator(self, data: Dict[str, Any]):
        """
        Creates iterator to iterate over all instances in data dictionary.
        Each iterator output element is a keyword argument dictionary with
        string keys.

        Each item key in the input data dictionary has to be of type string.
        - If the item value is not a sequence, the respective item key/value pair
          will be written to each iterator output dictionary.
        - If the item value is a sequence and the item key ends with '_batch',
          a check will be made to make sure length matches number of instances.
          The value of each instance in the sequence will be added to the respective
          iterator output dictionary with the '_batch' suffix removed.
        - If the item value is a sequence but doesn't end with '_batch', it will be treated
          as a simple value and the respective item key/value pair will be
          written to each iterator output dictionary.

        Parameters
        ----------
        data: dict[str, any]
            The data input dictionary.

        Returns
        -------
        iterator
            Each iterator output element is a keyword argument dictionary (string keys).

        """
        n_instances = len(data["x_batch"])

        for key, value in data.items():
            # If data-value is not a Sequence or a string, create list of repeated values with length of n_instances.
            if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, str):
                data[key] = [value for _ in range(n_instances)]

            # If data-value is a sequence and ends with '_batch', only check for correct length.
            elif key.endswith("_batch"):
                if len(value) != n_instances:
                    # Sequence has to have correct length.
                    raise ValueError(
                        f"'{key}' has incorrect length (expected: {n_instances}, is: {len(value)})"
                    )

            # If data-value is a sequence and doesn't end with '_batch', create
            # list of repeated sequences with length of n_instances.
            else:
                data[key] = [value for _ in range(n_instances)]

        # We create a list of dictionaries where each dictionary holds all data for a single instance.
        # We remove the '_batch' suffix if present.
        data_instances = [
            {
                re.sub("_batch", "", key): value[id_instance]
                for key, value in data.items()
            }
            for id_instance in range(n_instances)
        ]

        iterator = tqdm(
            enumerate(data_instances),
            total=n_instances,
            disable=not self.display_progressbar,  # Create progress bar if desired.
            desc=f"Evaluating {self.__class__.__name__}",
        )

        return iterator

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        **kwargs,
    ) -> Optional[Any]:
        """
        Implement this method if you need custom postprocessing of results or
        additional attributes.

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
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        kwargs: any, optional
            Additional data which was created in custom_preprocess().

        Returns
        -------
        any
            Can be implemented, optionally by the child class.
        """
        pass

    def plot(
        self,
        plot_func: Callable,
        show: bool = True,
        path_to_save: Union[str, None] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Basic plotting functionality for Metric class.
        The user provides a plot_func (Callable) that contains the actual plotting logic (but returns None).

        Parameters
        ----------
        plot_func: callable
            A Callable with the actual plotting logic.
        show: boolean
            A boolean to state if the plot shall be shown.
        path_to_save (str):
            A string that specifies the path to save file.
        args: optional
            An optional with additional arguments.
        kwargs: optional
            An optional dict with additional arguments.

        Returns
        -------
        None
        """
        # Get plotting func if not provided.
        if plot_func is None:
            plot_func = self.default_plot_func

        # Asserts.
        asserts.assert_plot_func(plot_func=plot_func)

        # Plot!
        plot_func(*args, **kwargs)

        if show:
            plt.show()

        if path_to_save:
            plt.savefig(fname=path_to_save, dpi=400)

        return None

    @property
    def interpret_scores(self) -> None:
        """
        Get an interpretation of the scores.
        """
        print(self.__init__.__doc__.split(".")[1].split("References")[0])

    @property
    def get_params(self) -> dict:
        """
        List parameters of metric.

        Returns
        -------
        dict
            A dictionary with attributes if not excluded from pre-determined list.
        """
        attr_exclude = [
            "args",
            "kwargs",
            "all_results",
            "last_results",
            "default_plot_func",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in attr_exclude}


class PerturbationMetric(Metric):
    """
    Implementation base PertubationMetric class.

    Metric categories such as Faithfulness and Robustness share certain characteristics when it comes to perturbations.
    As follows, this metric class is created which has additional attributes for perturbations.
    """

    @asserts.attributes_check
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

        # Save perturbation metric attributes.
        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        self.perturb_func_kwargs = perturb_func_kwargs

    @abstractmethod
    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
    ) -> Any:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        Any
        """
        raise NotImplementedError()
