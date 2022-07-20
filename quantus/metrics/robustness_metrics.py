"""This module contains the collection of robustness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import PerturbationMetric
from ..helpers import asserts
from ..helpers import similar_func
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.norm_func import fro_norm
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.perturb_func import gaussian_noise, uniform_noise, translation_x_direction


class LocalLipschitzEstimate(PerturbationMetric):
    """
    Implementation of the Local Lipschitz Estimated (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).

    """

    @attributes_check
    def __init__(
            self,
            similarity_func: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_numerator: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_denominator: Optional[Callable] = None,  # TODO: specify expected function signature
            nr_samples: int = 200,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            output_func: Optional[Callable] = None,  # TODO: specify expected function signature
            perturb_func: Callable = None,  # TODO: specify expected function signature
            perturb_mean: float = 0.0,
            perturb_std: float = 0.1,
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
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=lipschitz_constant.
        norm_numerator (callable): Function for norm calculations on the numerator.
            If None, the default value is used, default=distance_euclidean.
        norm_denominator (callable): Function for norm calculations on the denominator.
            If None, the default value is used, default=distance_euclidean.
        nr_samples (integer): The number of samples iterated, default=200.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=gaussian_noise.
        perturb_std (float): The amount of noise added, default=0.1.
        perturb_mean (float): The mean of noise added, default=0.0.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = gaussian_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['perturb_mean'] = perturb_mean
        perturb_func_kwargs['perturb_std'] = perturb_std

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
        if similarity_func is None:
            similarity_func = similar_func.lipschitz_constant
        self.similarity_func = similarity_func

        if norm_numerator is None:
            norm_numerator = similar_func.distance_euclidean
        self.norm_numerator = norm_numerator

        if norm_denominator is None:
            norm_denominator = similar_func.distance_euclidean
        self.norm_denominator = norm_denominator

        self.nr_samples = nr_samples

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'perturb_std', the number of samples iterated "
                    "over 'nr_samples', the function to perturb the input 'perturb_func',"
                    " the similarity metric 'similarity_func' as well as norm "
                    "calculations on the numerator and denominator of the lipschitz "
                    "equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'On the robustness of "
                    "interpretability methods.' arXiv preprint arXiv:1806.08049 (2018). and "
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint "
                    "arXiv:1806.07538 (2018)"
                ),
            )
            warn_func.warn_noise_zero(noise=perturb_std)

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
        >> metric = LocalLipschitzEstimate(abs=True, normalise=False)
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
        similarity_max = 0.0
        for i in range(self.nr_samples):

            # Perturb input.
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.perturb_func_kwargs,
            )
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x_input,
                targets=y,
                **self.explain_func_kwargs,
            )

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed, **self.normalise_func_kwargs,
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

                # Measure similarity.
                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed.flatten(),
                )
                similarity_max = max(similarity, similarity_max)

        return similarity_max

    def custom_preprocess(
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


class MaxSensitivity(PerturbationMetric):
    """
    Implementation of max-sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the maximum sensitivity is captured.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).
    """

    @attributes_check
    def __init__(
            self,
            similarity_func: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_numerator: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_denominator: Optional[Callable] = None,  # TODO: specify expected function signature
            nr_samples: int = 200,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            output_func: Optional[Callable] = None,  # TODO: specify expected function signature
            perturb_func: Callable = None,  # TODO: specify expected function signature
            lower_bound: float = 0.2,
            upper_bound: Optional[float] = None,
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
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=difference.
        norm_numerator (callable): Function for norm calculations on the numerator.
            If None, the default value is used, default=fro_norm
        norm_denominator (callable): Function for norm calculations on the denominator.
            If None, the default value is used, default=fro_norm
        nr_samples (integer): The number of samples iterated, default=200.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=uniform_noise.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        lower_bound (float): Lower Bound of Perturbation, default=0.2.
        upper_bound (None, float): Upper Bound of Perturbation, default=None.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = uniform_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['lower_bound'] = lower_bound
        perturb_func_kwargs['upper_bound'] = upper_bound

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
        if similarity_func is None:
            similarity_func = similar_func.difference
        self.similarity_func = similarity_func

        if norm_numerator is None:
            norm_numerator = fro_norm
        self.norm_numerator = norm_numerator

        if norm_denominator is None:
            norm_denominator = fro_norm
        self.norm_denominator = norm_denominator

        self.nr_samples = nr_samples

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn_func.warn_noise_zero(noise=lower_bound)

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
        >> metric = MaxSensitivity(abs=True, normalise=False)
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
        sensitivities_norm_max = 0.0
        for _ in range(self.nr_samples):

            # Perturb input.
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.perturb_func_kwargs,
            )
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x_input,
                targets=y,
                **self.explain_func_kwargs,
            )

            if self.normalise:
                a_perturbed = self.normalise_func(a_perturbed)

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            # Measure sensitivity.
            sensitivities = self.similarity_func(
                a=a.flatten(), b=a_perturbed.flatten()
            )
            numerator = self.norm_numerator(a=sensitivities)
            denominator  = self.norm_denominator(a=x.flatten())
            sensitivities_norm = numerator / denominator

            if sensitivities_norm > sensitivities_norm_max:
                sensitivities_norm_max = sensitivities_norm

        return sensitivities_norm_max

    def custom_preprocess(
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


class AvgSensitivity(PerturbationMetric):
    """
    Implementation of avg-sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the average sensitivity is captured.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    def __init__(
            self,
            similarity_func: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_numerator: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_denominator: Optional[Callable] = None,  # TODO: specify expected function signature
            nr_samples: int = 200,
            abs: bool = False,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            output_func: Optional[Callable] = None,  # TODO: specify expected function signature
            perturb_func: Callable = None,  # TODO: specify expected function signature
            lower_bound: float = 0.2,
            upper_bound: Optional[float] = None,
            softmax: bool = False,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            default_plot_func: Optional[Callable] = None,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=difference.
        norm_numerator (callable): Function for norm calculations on the numerator.
            If None, the default value is used, default=fro_norm
        norm_denominator (callable): Function for norm calculations on the denominator.
            If None, the default value is used, default=fro_norm
        nr_samples (integer): The number of samples iterated, default=200.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=uniform_noise.
        lower_bound (float): Lower Bound of Perturbation, default=0.2.
        upper_bound (None, float): Upper Bound of Perturbation, default=None.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = uniform_noise

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs['lower_bound'] = lower_bound
        perturb_func_kwargs['upper_bound'] = upper_bound

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
        if similarity_func is None:
            similarity_func = similar_func.difference
        self.similarity_func = similarity_func

        if norm_numerator is None:
            norm_numerator = fro_norm
        self.norm_numerator = norm_numerator

        if norm_denominator is None:
            norm_denominator = fro_norm
        self.norm_denominator = norm_denominator

        self.nr_samples = nr_samples

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn_func.warn_noise_zero(noise=lower_bound)

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
        >> metric = AvgSensitivity(abs=True, normalise=False)
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
        sub_results = [None for _ in range(self.nr_samples)]
        for sample_idx in range(self.nr_samples):

            # Perturb input.
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.perturb_func_kwargs,
            )
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Generate explanation based on perturbed input x.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x_input,
                targets=y,
                **self.explain_func_kwargs,
            )

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed, **self.normalise_func_kwargs
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            sensitivities = self.similarity_func(
                a=a.flatten(), b=a_perturbed.flatten()
            )
            sensitivities_numerator = self.norm_numerator(a=sensitivities)
            sensitivities_denominator = self.norm_denominator(a=x.flatten())
            sensitivities_norm = sensitivities_numerator / sensitivities_denominator

            sub_results[sample_idx] = sensitivities_norm

        # Append average sensitivity score.
        return float(np.mean(sub_results))


class Continuity(PerturbationMetric):
    """
    Implementation of the Continuity test by Montavon et al., 2018.

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Assumptions:
        - In this implementation, we assume that height and width dimensions are equally sized.
    """

    @attributes_check
    def __init__(
            self,
            similarity_func: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_numerator: Optional[Callable] = None,  # TODO: specify expected function signature
            norm_denominator: Optional[Callable] = None,  # TODO: specify expected function signature
            nr_samples: int = 200,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            output_func: Optional[Callable] = None,  # TODO: specify expected function signature
            perturb_func: Callable = None,  # TODO: specify expected function signature
            perturb_baseline: str = "black",
            patch_size: int = 7,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            nr_steps: int = 28,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=lipschitz_constant.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=translation_x_direction.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        patch_size (integer): The patch size for masking, default=7.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        nr_steps (integer): The number of steps to iterate over, default=28.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = translation_x_direction

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
        if similarity_func is None:
            similarity_func = similar_func.lipschitz_constant
        self.similarity_func = similarity_func

        self.patch_size = patch_size
        self.nr_steps = nr_steps

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "how many patches to split the input image to 'nr_patches', "
                    "the number of steps to iterate over 'nr_steps', the value to replace"
                    " the masking with 'perturb_baseline' and in what direction to "
                    "translate the image 'perturb_func'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73, 1-15 (2018"
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
    ) -> List[List[float]]:
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
        >> metric = Continuity(abs=True, normalise=False)
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
        sub_results = {k: [] for k in range(self.nr_patches + 1)}

        for step in range(self.nr_steps):

            # Generate explanation based on perturbed input x.
            dx_step = (step + 1) * self.dx
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                perturb_dx=dx_step,
                **self.perturb_func_kwargs,
            )
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)

            # Generate explanations on perturbed input.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x_input,
                targets=y,
                **self.explain_func_kwargs,
            )
            # Taking the first element, since a_perturbed will be expanded to a batch dimension
            # not expected by the current index management functions
            a_perturbed = utils.expand_attribution_channel(a_perturbed, x_input)[0]

            if self.normalise:
                a_perturbed = self.normalise_func(a_perturbed, **self.normalise_func_kwargs)

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            # Store the prediction score as the last element of the sub_self.last_results dictionary.
            y_pred = float(model.predict(x_input)[:, y])

            sub_results[self.nr_patches].append(y_pred)

            # create patches by splitting input into grid
            axis_iterators = [
                range(0, x_input.shape[axis], self.patch_size) for axis in self.a_axes
            ]
            for ix_patch, top_left_coords in enumerate(
                    itertools.product(*axis_iterators)
            ):

                # Create slice for patch.
                patch_slice = utils.create_patch_slice(
                    patch_size=self.patch_size,
                    coords=top_left_coords,
                )

                a_perturbed_patch = a_perturbed[
                    utils.expand_indices(a_perturbed, patch_slice, self.a_axes)
                ]

                if self.normalise:
                    a_perturbed_patch = self.normalise_func(
                        a_perturbed_patch.flatten()
                    )

                if self.abs:
                    a_perturbed_patch = np.abs(a_perturbed_patch.flatten())

                # Sum attributions for patch.
                patch_sum = float(sum(a_perturbed_patch))
                sub_results[ix_patch].append(patch_sum)

        return sub_results

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Infer attribution axes for perturbation function.
        self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

        # Get number of patches for input shape (ignore batch and channel dim).
        self.nr_patches = utils.get_nr_patches(
            patch_size=self.patch_size,
            shape=x_batch.shape[2:],
            overlap=True,
        )

        self.dx = np.prod(x_batch.shape[2:]) // self.nr_steps

        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

        # Asserts.
        asserts.assert_patch_size(patch_size=self.patch_size, shape=x_batch.shape[2:])

        return model, x_batch, y_batch, a_batch, s_batch

    @property
    def aggregated_score(self):
        """
        Implements a continuity correlation score (an addition to the original method) to evaluate the
        relationship between change in explanation and change in function output. It can be seen as an
        quantitative interpretation of visually determining how similar f(x) and R(x1) curves are.
        """
        return np.mean(
            [
                self.similarity_func(
                    self.last_results[sample][self.nr_patches],
                    self.last_results[sample][ix_patch],
                )
                for ix_patch in range(self.nr_patches)
                for sample in self.last_results.keys()
            ]
        )
