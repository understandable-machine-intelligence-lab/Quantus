"""This module implements the base class for creating evaluation metrics."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..helpers import utils
from ..helpers import asserts
from ..helpers.model_interface import ModelInterface
from ..helpers import warn_func


class Metric:
    """
    Implementation of the base Metric class.
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
        Initialise the Metric base class.

        Each of the defined metrics in Quantus, inherits from Metric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply () on each
                      instance and finally call custom_preprocess().
                      To use this method the child Metric needs to implement ().
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

        # Run deprecation warnings.
        warn_func.deprecation_warnings(kwargs)
        warn_func.check_kwargs(kwargs)

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

        self.a_axes = None

        self.last_results = []
        self.all_results = []

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
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
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
        warn_func.deprecation_warnings(kwargs)
        warn_func.check_kwargs(kwargs)

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
            custom_batch=custom_batch,
        )

        # Create progress bar if desired.
        iterator = tqdm(
            enumerate(
                zip(
                    x_batch,
                    y_batch,
                    a_batch,
                    s_batch,
                    custom_batch,
                    custom_preprocess_batch,
                )
            ),
            total=len(x_batch),
            disable=not self.display_progressbar,
            desc=f"Evaluating {self.__class__.__name__}",
        )
        self.last_results = [None for _ in x_batch]
        for id_instance, (
            x_instance,
            y_instance,
            a_instance,
            s_instance,
            c_instance,
            p_instance,
        ) in iterator:
            result = self.evaluate_instance(
                i=int(id_instance),
                model=model,
                x=x_instance,
                y=y_instance,
                a=a_instance,
                s=s_instance,
                c=c_instance,
                p=p_instance,
            )
            self.last_results[id_instance] = result

        # Call custom post-processing.
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
        )

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
        i: int,
        model: ModelInterface,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
    ) -> Any:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        i: integer
            The evaluation instance.
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
        c: any
            The custom input to be evaluated on an instance-basis.
        p: any
            The custom preprocess input to be evaluated on an instance-basis.
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
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: bool,
        device: Optional[str],
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:
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

        model: Union[torch.nn.Module, tf.keras.Model]
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
        kwargs: optional
            Keyword arguments.

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

        # Call custom pre-processing from inheriting class.
        (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        ) = self.custom_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
        )

        # Normalise with specified keyword arguments if requested.
        if self.normalise:
            a_batch = self.normalise_func(
                a=a_batch,
                normalized_axes=list(range(np.ndim(a_batch)))[1:],
                **self.normalise_func_kwargs,
            )

        # Take absolute if requested.
        if self.abs:
            a_batch = np.abs(a_batch)

        # This is needed for iterator (zipped over x_batch, y_batch, a_batch, s_batch, custom_batch).
        if s_batch is None:
            s_batch = [None for _ in x_batch]
        if custom_batch is None:
            custom_batch = [None for _ in x_batch]

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:
        """
        Implement this method if you need custom preprocessing of data,
        model alteration or simply for creating/initialising additional attributes.

        If this function is implemented for your metric, make sure to add in the metric-specific
        function custom_preprocess():

        >> custom_preprocess_batch = [None for _ in x_batch]

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
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
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        tuple
            In addition to the x_batch, y_batch, a_batch, s_batch and custom_batch,
            returning a custom preprocess batch (custom_preprocess_batch).
        """
        custom_preprocess_batch = [None for _ in x_batch]
        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Optional[Any]:
        """
        Implement this method if you need custom postprocessing of results or
        additional attributes.

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
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
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        any
            Can be implemented, optionally by the user.
        """
        pass

    def plot(
        self,
        plot_func: Union[Callable, None] = None,
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
            "disable_warnings",
            "display_progressbar",
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

        self.perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}

        self.perturb_func_kwargs = perturb_func_kwargs
