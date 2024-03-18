"""This module implements the base class for creating evaluation metrics."""
# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

import functools
import logging
import os
import sys
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Sequence,
    Set,
    TypeVar,
    Optional,
    Union,
    TYPE_CHECKING,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import gen_batches
from tqdm.auto import tqdm

from quantus.helpers import asserts, utils, warn
from quantus.functions.normalise_func import normalise_by_max
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final

if TYPE_CHECKING:
    import keras
    from torch import nn

D = TypeVar("D", bound=Dict[str, Any])
log = logging.getLogger(__name__)

# Return value of __call__
R = TypeVar("R")


class Metric(Generic[R]):
    """
    Interface defining Metrics' API.
    """

    # Class attributes.
    name: ClassVar[str]
    data_applicability: ClassVar[Set[DataType]]
    model_applicability: ClassVar[Set[ModelType]]
    score_direction: ClassVar[ScoreDirection]
    evaluation_category: ClassVar[EvaluationCategory]

    # Instance attributes.
    explain_func: Callable
    explain_func_kwargs: Dict[str, Any]
    a_axes: Sequence[int]
    evaluation_scores: Any
    all_evaluation_scores: Any
    normalise_func: Optional[Callable[[np.ndarray], np.ndarray]]

    def __init__(
        self,
        abs: bool,
        normalise: bool,
        normalise_func: Optional[Callable],
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

        The content of evaluation_scores will be appended to all_evaluation_scores (list) at the end of
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

        if aggregate_func is None:
            aggregate_func = np.mean

        if normalise_func is None:
            normalise_func = normalise_by_max

        if normalise_func_kwargs is not None:
            normalise_func = functools.partial(normalise_func, **normalise_func_kwargs)

        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        self.abs = abs
        self.normalise = normalise
        self.return_aggregate = return_aggregate
        self.aggregate_func = aggregate_func
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs or {}

        self.default_plot_func = default_plot_func

        # We need underscores here to avoid conflict with @property descriptor.
        self._disable_warnings = disable_warnings
        self._display_progressbar = display_progressbar

        self.a_axes = None

        self.evaluation_scores = []
        self.all_evaluation_scores = []

    @no_type_check
    def __call__(
        self,
        model: Union[keras.Model, nn.Module, None],
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Any = None,
        **kwargs,
    ) -> R:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        The content of evaluation_scores will be appended to all_evaluation_scores (list) at the end of
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
        evaluation_scores: list
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        data: Dict[str, Any] = self.general_preprocess(
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

        self.evaluation_scores = []
        for d_ix, data_batch in enumerate(batch_generator):
            data_batch = self.batch_preprocess(data_batch)
            result = self.evaluate_batch(**data_batch)
            self.evaluation_scores.extend(result)

        # Call post-processing.
        self.custom_postprocess(**data)

        if self.return_aggregate:
            if self.aggregate_func:
                try:
                    self.evaluation_scores = [
                        self.aggregate_func(self.evaluation_scores)
                    ]
                except Exception as ex:
                    log.error(
                        f"The aggregation of evaluation scores failed with {ex}. Check that "
                        "'aggregate_func' supplied is appropriate for the data "
                        "in 'evaluation_scores'."
                    )
            else:
                raise KeyError(
                    "Specify an 'aggregate_func' (Callable) to aggregate evaluation scores."
                )

        # Append the content of the last results to all results.
        self.all_evaluation_scores.extend(self.evaluation_scores)

        return self.evaluation_scores  # type: ignore

    @abstractmethod
    @no_type_check
    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: Optional[np.ndarray],
        **kwargs,
    ) -> R:
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

    @final
    def general_preprocess(
        self,
        model: Union[keras.Model, nn.Module, None],
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Callable,
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict[str, Any]],
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

        if model is not None:
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
        self.explain_func_kwargs = explain_func_kwargs or {}

        # Include device in explain_func_kwargs.
        if device is not None and "device" not in self.explain_func_kwargs:
            self.explain_func_kwargs["device"] = device

        if a_batch is not None:
            a_batch = utils.expand_attribution_channel(a_batch, x_batch)
            asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)
            self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

            # Normalise with specified keyword arguments if requested.
            if self.normalise:
                a_batch = self.normalise_func(a_batch)

            # Take absolute if requested.
            if self.abs:
                a_batch = np.abs(a_batch)

        else:
            # If no explanations provided, we will compute them on batch-level to avoid OOM.
            asserts.assert_explain_func(explain_func=self.explain_func)

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
        if custom_preprocess_dict is not None:
            data.update(custom_preprocess_dict)

        # Remove custom_batch if not used.
        if data["custom_batch"] is None:
            del data["custom_batch"]

        return data

    def custom_preprocess(
        self,
        *,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        custom_batch: Any,
        **kwargs,
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
        kwargs:
            Optional, metric-specific parameters.


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

    def custom_postprocess(
        self,
        *,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        **kwargs,
    ):
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
        any:
            Can be implemented, optionally by the child class.
        """
        pass

    @final
    def generate_batches(
        self,
        data: D,
        batch_size: int,
    ) -> Generator[D, None, None]:
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
        iterator:
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

        n_batches = np.ceil(n_instances / batch_size)

        with tqdm(total=n_batches, disable=not self.display_progressbar) as pbar:
            for batch_idx in gen_batches(n_instances, batch_size):
                batch = {
                    key: value[batch_idx.start : batch_idx.stop]
                    for key, value in batched_value_kwargs.items()
                }
                # Yield batch dictionary including single value keyword arguments.
                yield {**batch, **single_value_kwargs}
                # Update progressbar by number of samples in this batch.
                pbar.update(batch_idx.stop - batch_idx.start)

    def plot(
        self,
        plot_func: Optional[Callable] = None,
        show: bool = True,
        path_to_save: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Basic plotting functionality for Metric class.
        The user provides a plot_func (Callable) that contains the actual plotting logic (but returns None).

        Parameters
        ----------
        plot_func: callable
            A Callable with the actual plotting logic. Default set to None, which implies default_plot_func is set.
        show: boolean
            A boolean to state if the plot shall be shown.
        path_to_save: (str)
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

    def interpret_scores(self):
        """Get an interpretation of the scores."""
        print(self.__init__.__doc__.split(".")[1].split("References")[0])

    @property
    def get_params(self) -> Dict[str, Any]:
        """
        List parameters of metric.

        Returns
        -------
        dict:
            A dictionary with attributes if not excluded from pre-determined list.
        """
        attr_exclude = [
            "args",
            "kwargs",
            "all_evaluation_scores",
            "evaluation_scores",
            "default_plot_func",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in attr_exclude}

    @final
    def batch_preprocess(self, data_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        If `data_batch` has no `a_batch`, will compute explanations.
        This needs to be done on batch level to avoid OOM. Additionally will set `a_axes` property if it is None,
        this can be done earliest after we have first `a_batch`.

        Parameters
        ----------
        data_batch:
            A single entry yielded from the generator return by `self.generate_batches(...)`

        Returns
        -------
        data_batch:
            Dictionary, which is ready to be passed down to `self.evaluate_batch`.
        """

        x_batch = data_batch["x_batch"]

        a_batch = data_batch.get("a_batch")

        if a_batch is None:
            # Generate batch of explanations lazily, so we don't hit OOM
            model = data_batch["model"]
            y_batch = data_batch["y_batch"]
            a_batch = self.explain_batch(model, x_batch, y_batch)
            data_batch["a_batch"] = a_batch

        if self.a_axes is None:
            self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

        custom_batch = self.custom_batch_preprocess(**data_batch)
        if custom_batch is not None:
            data_batch.update(custom_batch)
        return data_batch

    def custom_batch_preprocess(
        self,
        *,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Implement this method if you need custom preprocessing of data
        or simply for creating/initialising additional attributes or assertions
        before a `data_batch` can be evaluated.

        Parameters
        ----------
        model:
            A model that is subject to explanation.
        x_batch:
            A np.ndarray which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.
        a_batch:
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        kwargs:
            Optional, metric-specific parameters.

        Returns
        -------
        dict:
            Optional dictionary with additional kwargs, which will be passed to `self.evaluate_batch(...)`
        """
        pass

    @final
    def explain_batch(
        self,
        model: Union[ModelInterface, keras.Model, nn.Module],
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Compute explanations, normalise and take absolute (if was configured so during metric initialization.)
        This method should primarily be used if you need to generate additional explanation
        in metrics body. It encapsulates typical for Quantus pre- and postprocessing approach.
        It will do few things:
            - call model.shape_input (if ModelInterface instance was provided)
            - unwrap model (if ModelInterface instance was provided)
            - call explain_func
            - expand attribution channel
            - (optionally) normalise a_batch
            - (optionally) take np.abs of a_batch


        Parameters
        -------
        model:
            A model that is subject to explanation.
        x_batch:
            A np.ndarray which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.

        Returns
        -------
        a_batch:
            Batch of explanations ready to be evaluated.
        """

        if isinstance(model, ModelInterface):
            # Sometimes the model is our wrapper, but sometimes raw Keras/Torch model.
            x_batch = model.shape_input(
                x=x_batch,
                shape=x_batch.shape,
                channel_first=True,
                batched=True,
            )
            model = model.get_model()

        a_batch = self.explain_func(
            model=model, inputs=x_batch, targets=y_batch, **self.explain_func_kwargs
        )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch)
        asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

        # Normalise and take absolute values of the attributions, if configured during metric instantiation.
        if self.normalise:
            a_batch = self.normalise_func(a_batch)

        if self.abs:
            a_batch = np.abs(a_batch)

        return a_batch

    @property
    def display_progressbar(self) -> bool:
        """A helper to avoid polluting test outputs with tqdm progress bars."""
        return (
            self._display_progressbar
            and
            # Don't show progress bar in github actions.
            "GITHUB_ACTIONS" not in os.environ
            and
            # Don't show progress bar when running unit tests.
            "PYTEST" not in os.environ
        )

    @property
    def disable_warnings(self) -> bool:
        """A helper to avoid polluting test outputs with warnings."""
        return (
            self._disable_warnings
            # Don't show progress bar in github actions.
            or "GITHUB_ACTIONS" not in os.environ
            # Don't show progress bar when running unit tests.
            or "PYTEST" in os.environ
        )
