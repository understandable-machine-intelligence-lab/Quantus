"""This module contains the collection of complexity metrics to evaluate attribution-based explanations of neural network models."""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative


class Sparseness(Metric):
    """
    Implementation of Sparseness metric by Chalasani et al., 2020.

    Sparseness is quantified using the Gini Index applied to the vector of the absolute values of attributions. The
    test asks that features that are truly predictive of the output F(x) should have significant contributions, and
    similarly, that irrelevant (or weakly-relevant) features should have negligible contributions.

    References:
        1) Chalasani, Prasad, et al. "Concise explanations of neural networks using adversarial training."
        International Conference on Machine Learning. PMLR, 2020.

    Assumptions:
        Based on authors' implementation as found on the following link:
        https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py.

    """

    @attributes_check
    def __init__(
            self,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,  # TODO: specify expected function input/output
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        default_plot_func (callable): Callable that plots the metrics result.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if not abs:
            # TODO: document this behaviour
            abs = True
            warn_func.warn_absolutes_applied()

        if normalise_func is None:
            normalise_func = normalise_by_negative

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

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "normalising 'normalise' (and 'normalise_func') and if taking absolute"
                    " values of attributions 'abs'"
                ),
                citation=(
                    "Chalasani, Prasad, et al. Concise explanations of neural networks using "
                    "adversarial training.' International Conference on Machine Learning. PMLR, "
                    "(2020)"
                ),
            )

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            channel_first: Optional[bool] = None,
            explain_func: Optional[Callable] = None,  # Specify function signature
            explain_func_kwargs: Optional[Dict[str, Any]] = None,
            model_predict_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None,
            **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None, default=None.
        explain_func (callable): Callable generating attributions, default=Callable.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}

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
        >> metric = Sparseness(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):
        a = np.array(
            np.reshape(a, (np.prod(x.shape[1:]),)),
            dtype=np.float64,
        )
        a += 0.0000001
        a = np.sort(a)
        score = (
            (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a))
            / (a.shape[0] * np.sum(a))
        )
        return score


class Complexity(Metric):
    """
    Implementation of Complexity metric by Bhatt et al., 2020.

    Complexity of attributions is defined as the entropy of the fractional contribution of feature x_i to the total
    magnitude of the attribution. A complex explanation is one that uses all features in its explanation to explain
    some decision. Even though such an explanation may be faithful to the model output, if the number of features is
    too large it may be too difficult for the user to understand the explanations, rendering it useless.

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    @attributes_check
    def __init__(
            self,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,  # TODO: specify expected function input/output
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        default_plot_func (callable): Callable that plots the metrics result.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if not abs:
            # TODO: document this behaviour
            abs = True
            warn_func.warn_absolutes_applied()

        if normalise_func is None:
            normalise_func = normalise_by_negative

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

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "normalising 'normalise' (and 'normalise_func') and if taking absolute"
                    " values of attributions 'abs'"
                ),
                citation=(
                    "Bhatt, Umang, Adrian Weller, and José MF Moura. 'Evaluating and aggregating"
                    " feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)"
                ),
            )

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            channel_first: Optional[bool] = None,
            explain_func: Optional[Callable] = None,  # Specify function signature
            explain_func_kwargs: Optional[Dict[str, Any]] = None,
            model_predict_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None,
            **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None, default=None.
        explain_func (callable): Callable generating attributions, default=Callable.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}

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
        >> metric = Complexity(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):
        a = np.array(
            np.reshape(a, (np.prod(x.shape[1:]),)),
            dtype=np.float64,
        ) / np.sum(np.abs(a))

        return scipy.stats.entropy(pk=a)


class EffectiveComplexity(Metric):
    """
    Implementation of Effective complexity metric by Nguyen at el., 2020.

    Effective complexity measures how many attributions in absolute values are exceeding a certain threshold (eps)
    where a value above the specified threshold implies that the features are important and under indicates it is not.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
    """

    @attributes_check
    def __init__(
            self,
            eps: float = 1e-5,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,  # TODO: specify expected function input/output
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        eps (float): Attributions threshold, default=1e-5.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        default_plot_func (callable): Callable that plots the metrics result.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if not abs:
            # TODO: document this behaviour
            abs = True
            warn_func.warn_absolutes_applied()

        if normalise_func is None:
            normalise_func = normalise_by_negative

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

        # Save metric-specific attributes.
        self.eps = eps

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "normalising 'normalise' (and 'normalise_func') and if taking absolute"
                    " values of attributions 'abs' and the choice of threshold 'eps'"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)."
                ),
            )

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            channel_first: Optional[bool] = None,
            explain_func: Optional[Callable] = None,  # Specify function signature
            explain_func_kwargs: Optional[Dict[str, Any]] = None,
            model_predict_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None,
            **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is subject to explanation
        x_batch: a np.ndarray which contains the input data that are explained
        y_batch: a np.ndarray which contains the output labels that are explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        channel_first (boolean, optional): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None, default=None.
        explain_func (callable): Callable generating attributions, default=Callable.
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}

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
        >> metric = EffectiveComplexity(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):
        a = a.flatten()
        return int(np.sum(a > self.eps))  # casting to int needed?
