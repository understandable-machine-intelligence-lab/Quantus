"""This module contains the collection of axiomatic metrics to evaluate attribution-based explanations of neural network models."""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import Metric, PerturbationMetric
from ..helpers import asserts
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.perturb_func import baseline_replacement_by_indices
from ..helpers.perturb_func import baseline_replacement_by_shift


class Completeness(PerturbationMetric):
    """
    Implementation of Completeness test by Sundararajan et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018.

    Attribution completeness asks that the total attribution is proportional to the explainable
    evidence at the output/ or some function of the model output. Or, that the attributions
    add up to the difference between the model output F at the input x and the baseline b.

    References:
        1) Completeness - Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks."
        International Conference on Machine Learning. PMLR, 2017.
        2) Summation to delta - Shrikumar, Avanti, Peyton Greenside, and Anshul Kundaje. "Learning important
        features through propagating activation differences." International Conference on Machine Learning. PMLR, 2017.
        3) Conservation - Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting
        and understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Assumptions:
        This implementation does completeness test against logits, not softmax.
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            output_func: Optional[Callable] = None,  # TODO: specify expected function signature
            perturb_baseline: str = "black",
            perturb_func: Callable = None,  # TODO: specify expected function signature
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        output_func (callable): Function applied to the difference between the model output at the input and the
            baseline before metric calculation. If output_func=None, the default value is used, default=lambda x: x.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['perturb_baseline'] = perturb_baseline

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            softmax=softmax,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        if output_func is None:
            output_func = lambda x: x
        self.output_func = output_func

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the function to modify the "
                    "model response 'output_func'"
                ),
                citation=(
                    "Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 'Axiomatic attribution for "
                    "deep networks.' International Conference on Machine Learning. PMLR, (2017)."
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
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}.
        kwargs: Keyword arguments (optional)

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
        >> metric = Completeness(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
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

    def process_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):

        x_baseline = self.perturb_func(
            arr=x,
            indices=np.arange(0, x.size),
            indexed_axes=np.arange(0, x.ndim),
            **self.perturb_func_kwargs,
        )

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        # Predict on baseline.
        x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
        y_pred_baseline = float(model.predict(x_input)[:, y])

        if np.sum(a) == self.output_func(y_pred - y_pred_baseline):
            return True
        else:
            return False


class NonSensitivity(PerturbationMetric):
    """
    Implementation of NonSensitivity by Nguyen at el., 2020.

    Non-sensitivity measures if zero-importance is only assigned to features, that the model is not
    functionally dependent on.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Ancona, Marco, et al. "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley
        Values Approximation." arXiv preprint arXiv:1903.10992 (2019).
        3) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    """

    @attributes_check
    def __init__(
            self,
            eps: float = 1e-5,
            n_samples: int = 100,
            features_in_step: int = 1,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            perturb_baseline: str = "black",
            perturb_func: Callable = None,  # TODO: specify expected function signature
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = True,
            default_plot_func: Optional[Callable] = None,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        eps (float): Attributions threshold, default=1e-5.
        n_samples (integer): The number of samples to iterate over, default=100.
        features_in_step (integer): The step size, default=1.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=True.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """

        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['perturb_baseline'] = perturb_baseline

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            softmax=softmax,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.eps = eps
        self.n_samples = n_samples
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the number of samples to iterate"
                    " over 'n_samples' and the threshold value function for the feature"
                    " to be considered having an insignificant contribution to the model"
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
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}.
        kwargs: Keyword arguments (optional)

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
        >> metric = NonSensitivity(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
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

    def process_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):
            a = a.flatten()

            non_features = set(list(np.argwhere(a).flatten() < self.eps))

            vars = []
            for i_ix, a_ix in enumerate(a[:: self.features_in_step]):

                preds = []
                a_ix = a[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ].astype(int)

                for _ in range(self.n_samples):
                    # Perturb input by indices of attributions.
                    x_perturbed = self.perturb_func(
                        arr=x,
                        indices=a_ix,
                        indexed_axes=self.a_axes,
                        **self.perturb_func_kwargs,
                    )

                    # Predict on perturbed input x.
                    x_input = model.shape_input(
                        x_perturbed, x.shape, channel_first=True
                    )
                    y_pred_perturbed = float(
                        model.predict(x_input)[:, y]
                    )
                    preds.append(y_pred_perturbed)

                    vars.append(np.var(preds))

            non_features_vars = set(list(np.argwhere(vars).flatten() < self.eps))

            return len(non_features_vars.symmetric_difference(non_features))

    def preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Infer attribution axes for perturbation function.
        self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch


class InputInvariance(PerturbationMetric):
    """
    Implementation of Completeness test by Kindermans et al., 2017.

    To test for input invaraince, we add a constant shift to the input data and then measure the effect
    on the attributions, the expectation is that if the model show no response, then the explanations should not.

    References:
        Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., Dähne Sven,
        Erhan Dumitru and Kim Been. "THE (UN)RELIABILITY OF SALIENCY METHODS" Article (2017).
    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            input_shift: int = -1,
            perturb_func: Callable = None,  # TODO: specify expected function signature
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        input_shift (integer): Shift to the input data, default=-1.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise:
            # TODO: document this behaviour
            normalise = False
            warn_func.warn_normalisation_skipped()

        if abs:
            # TODO: document this behaviour
            abs = False
            warn_func.warn_absolutes_skipped()

        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_shift

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['input_shift'] = input_shift

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
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
                sensitive_params=("input shift 'input_shift'"),
                citation=(
                    "Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., "
                    "Dähne Sven, Erhan Dumitru and Kim Been. 'THE (UN)RELIABILITY OF SALIENCY METHODS' Article (2017)."
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
        explain_func_kwargs (dict, optional): Keyword arguments to be passed to explain_func on call, default = {}.
        device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.
        model_predict_kwargs (dict, optional): Keyword arguments to be passed to the model's predict method, default = {}.
        kwargs: Keyword arguments (optional)

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
        >> metric = InputInvariance(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
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

    def process_instance(
            self,
            model: ModelInterface,
            x: np.ndarray,
            y: np.ndarray,
            a: np.ndarray,
            s: np.ndarray,
    ):

        x_shifted = self.perturb_func(
            arr=x,
            indices=np.arange(0, x.size),
            indexed_axes=np.arange(0, x.ndim),
            **self.perturb_func_kwargs,
        )
        x_shifted = model.shape_input(x_shifted, x.shape, channel_first=True)
        asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_shifted)

        # Generate explanation based on shifted input x.
        a_shifted = self.explain_func(
            model=model.get_model(), inputs=x_shifted, targets=y,
            **self.explain_func_kwargs
        )

        # Check if explanation of shifted input is similar to original.
        if (a.flatten() != a_shifted.flatten()).all():
            return True
        else:
            return False

    def preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

        return model, x_batch, y_batch, a_batch, s_batch
