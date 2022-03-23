"""This module implements the base class for creating evaluation measures."""
import warnings
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..helpers import utils
from ..helpers import asserts
from ..helpers import normalise_func
from ..helpers import warn_func


class Metric:
    """
    Implementation base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            warn_parametrisation_kwargs: Optional[Dict[str, str]] = None,
            **kwargs):
        """
        Initialise the Metric base class.

        Each of the defined metrics in Quantus, inherits from Metric base class.

        To add a new metric classes to the library, the minimum set of attributes that should be included are:

            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: Callable that make a normalising transformation of the attributions
            default_plot_func: Callable that plots the metrics result
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance

        """
        self.kwargs = kwargs
        self.abs = abs
        self.normalise = normalise

        if normalise_func is None:
            normalise_func = normalise_func.normalise_by_negative
        self.normalise_func = normalise_func
        if normalise_func_kwargs is None:
            normalise_func_kwargs = {}
        # TODO: deprecate this kind of unspecific kwargs passing
        # this code prioritizes normalise_func_kwargs items over kwargs items
        normalise_func_kwargs = {**kwargs, **normalise_func_kwargs}
        self.normalise_func_kwargs = normalise_func_kwargs

        self.perturb_func = perturb_func
        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        # TODO: deprecate this kind of unspecific kwargs passing
        # this code prioritizes perturb_func_kwargs items over kwargs items
        perturb_func_kwargs = {**kwargs, **perturb_func_kwargs}
        self.perturb_func_kwargs = perturb_func_kwargs

        self.default_plot_func = plot_func
        self.display_progressbar = display_progressbar
        self.disable_warnings = disable_warnings

        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            if warn_parametrisation_kwargs is None:
                raise ValueError(
                    "no warn_parametrisation_kwargs passed, but warnings are not disabled."
                )
            warn_func.warn_parameterisation(**warn_parametrisation_kwargs)


    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: optional args
            kwargs: optional dict

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            batch_size: int = 64,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: optional args
            kwargs: optional dict

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            self.channel_first = kwargs.get("channel_first")
        else:
            self.channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, self.channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first=self.channel_first)

        # get optional kwargs for model.predict call
        if model_predict_kwargs is None:
            model_predict_kwargs = {}
        # TODO: deprecate this kind of unspecific kwargs passing
        # this code prioritizes model_predict_kwargs items over kwargs items
        model_predict_kwargs = {**kwargs, **model_predict_kwargs}

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        if a_batch is None:

            # Asserts.
            asserts.assert_explain_func(explain_func=explain_func)

            if explain_func_kwargs is None:
                explain_func_kwargs = {}
            # TODO: deprecate this kind of unspecific kwargs passing
            # this code prioritizes explain_func_kwargs items over kwargs items
            explain_func_kwargs = {**kwargs, **explain_func_kwargs}

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **explain_func_kwargs,
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # TODO: maybe find better names instead of capital letters?
        # also: use these as argument names and deprecate _batch suffixes
        X = x_batch_s
        Y = y_batch
        A = a_batch
        S = s_batch

        # Asserts.
        asserts.assert_attributions(x_batch=X, a_batch=A)

        # Call pre-processing
        self.preprocess(X=X, Y=Y, A=A, S=S)

        if self.abs:
            # inplace execution
            np.abs(A, out=A)

        if self.normalise:
            # inplace execution
            normalise_func.normalise_batch(
                arr=A,
                normalise_func=self.normalise_func,
                normalise_func_kwargs=self.normalise_func_kwargs,
            )

        # create generator for generating batches
        batch_generator = utils.get_batch_generator(X, Y, A, batch_size=batch_size)

        # use tqdm progressbar if not disabled
        n_batches = utils.get_number_of_batches(X, batch_size=batch_size)
        iterator = tqdm(
            batch_generator,
            total=n_batches,
            disable=not self.display_progressbar,
        )

        for x_batch, y_batch, a_batch in iterator:
            result = self.process_batch(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                perturb_func=self.perturb_func,
                perturb_func_kwargs=self.perturb_func_kwargs,
                model_predict_kwargs=model_predict_kwargs,
            )
            self.last_results.append(result)

        # Call post-processing
        self.postprocess()

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


    def preprocess(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            S: np.ndarray,
    ):
        pass

    def postprocess(self):
        pass

    @property
    def interpret_scores(self) -> None:
        """

        Returns
        -------

        """
        print(self.__init__.__doc__.split(".")[1].split("References")[0])
        # print(self.__call__.__doc__.split("callable.")[1].split("Parameters")[0])

    @property
    def get_params(self) -> dict:
        """
        List parameters of metric.
        Returns: a dictionary with attributes if not excluded from pre-determined list
        -------

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

    def set_params(self, key: str, value: Any) -> dict:
        """
        Set a parameter of a metric.
        Parameters
        ----------
        key: attribute of metric to mutate
        value: value to update the key with

        -------
        Returns: the updated dictionary.

        """
        self.kwargs[key] = value
        return self.kwargs

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
        plot_func: a Callable with the actual plotting logic.
        show: a boolean to state if the plot shall be shown.
        path_to_save: a string that specifies the path to save file.
        args: an optional with additional arguments.
        kwargs: an optional dict with additional arguments.

        Returns: None.
        -------

        """

        # Get plotting func if not provided.
        if plot_func is None:
            plot_func =  self.default_plot_func

        # Asserts.
        asserts.assert_plot_func(plot_func=plot_func)

        # Plot!
        plot_func(*args, **kwargs)

        if show:
            plt.show()

        if path_to_save:
            plt.savefig(fname=path_to_save, dpi=400)

        return None
