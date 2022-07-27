"""This module contains the collection of faithfulness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import PerturbationMetric
from ..helpers import asserts
from ..helpers import plotting
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.similar_func import correlation_pearson, correlation_spearman, mse
from ..helpers.perturb_func import baseline_replacement_by_indices
from ..helpers.perturb_func import noisy_linear_imputation


class FaithfulnessCorrelation(PerturbationMetric):
    """
    Implementation of faithfulness correlation by Bhatt et al., 2020.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating feature-based model
        explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    @attributes_check
    def __init__(
        self,
        similarity_func: Optional[
            Callable
        ] = None,  # TODO: specify expected function signature
        nr_runs: int = 100,
        subset_size: int = 224,
        return_aggregate: bool = True,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
            If None, the default value is used, default=correlation_pearson.
        nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
        subset_size (integer): The size of subset, default=224.
        return_aggregate (boolean): Indicates whether an aggregated(mean) metric is returned, default=True.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
            similarity_func = correlation_pearson
        self.similarity_func = similarity_func

        self.nr_runs = nr_runs
        self.subset_size = subset_size
        self.return_aggregate = return_aggregate

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', size of subset |S| 'subset_size'"
                    " and the number of runs (for each input and explanation pair) "
                    "'nr_runs'"
                ),
                citation=(
                    "Bhatt, Umang, Adrian Weller, and José MF Moura. 'Evaluating and aggregating "
                    "feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = FaithfulnessCorrelation(abs=False, normalise=False)
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
            softmax=softmax,
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

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        pred_deltas = []
        att_sums = []

        # For each test data point, execute a couple of runs.
        for i_ix in range(self.nr_runs):

            # Randomly mask by subset size.
            a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            pred_deltas.append(float(y_pred - y_pred_perturb))

            # Sum attributions of the random subset.
            att_sums.append(np.sum(a[a_ix]))

        similarity = self.similarity_func(a=att_sums, b=pred_deltas)
        return similarity

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

        # Asserts.
        asserts.assert_value_smaller_than_input_size(
            x=x_batch, value=self.subset_size, value_name="subset_size"
        )

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.return_aggregate:
            self.last_results = [np.mean(self.last_results)]


class FaithfulnessEstimate(PerturbationMetric):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
    """

    @attributes_check
    def __init__(
        self,
        similarity_func: Optional[
            Callable
        ] = None,  # TODO: specify expected function signature
        nr_runs: int = 100,
        subset_size: int = 224,
        features_in_step: int = 1,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
            If None, the default value is used, default=correlation_spearman.
        nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
        subset_size (integer): The size of subset, default=224.
        features_in_step (integer): The size of the step, default=1.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
            similarity_func = correlation_spearman
        self.similarity_func = similarity_func

        self.nr_runs = nr_runs
        self.subset_size = subset_size
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and similarity function "
                    "'similarity_func'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint arXiv:1806.07538 (2018)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = FaithfulnessEstimate(abs=True, normalise=False)
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
            softmax=softmax,
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

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        pred_deltas = [None for _ in range(n_perturbations)]
        att_sums = [None for _ in range(n_perturbations)]

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            pred_deltas[i_ix] = float(y_pred - y_pred_perturb)

            # Sum attributions.
            att_sums[i_ix] = a[a_ix].sum()

        similarity = self.similarity_func(a=att_sums, b=pred_deltas)
        return similarity

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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch


class IterativeRemovalOfFeatures(PerturbationMetric):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    References:
        1) Rieger, Laura, and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

    """

    @attributes_check
    def __init__(
        self,
        segmentation_method: str = "slic",
        return_aggregate: bool = True,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "mean",
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
        segmentation_method (string): Image segmentation method:'slic' or 'felzenszwalb', default="slic".
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=correlation_pearson.
        nr_runs (integer): The number of runs (for each input and explanation pair), default=100.
        subset_size (integer): The size of subset, default=224.
        return_aggregate (boolean): Indicates whether an aggregated(mean) metric is returned, default=True.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
        self.segmentation_method = segmentation_method
        self.return_aggregate = return_aggregate

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated hyperparameters)"
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = IROF(abs=False, normalise=False)
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
            softmax=softmax,
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
        # Predict on x.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=self.segmentation_method,
        )
        nr_segments = segments.max()
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        preds = []

        for i_ix, s_ix in enumerate(s_indices):

            # Perturb input by indices of attributions.
            a_ix = np.nonzero(
                np.repeat((segments == s_ix).flatten(), self.nr_channels)
            )[0]

            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            # Normalise the scores to be within range [0, 1].
            preds.append(float(y_pred_perturb / y_pred))

        score = len(preds) - utils.calculate_auc(np.array(preds))
        return score

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Infer number of input channels and attribution axes for perturbation function.
        self.nr_channels = x_batch.shape[1]
        self.a_axes = utils.infer_attribution_axes(a_batch, x_batch)

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.return_aggregate:
            self.last_results = [np.mean(self.last_results)]

    @property
    def get_aggregated_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return [np.mean(results) for results in self.all_results]


class MonotonicityArya(PerturbationMetric):
    """
    Implementation of Montonicity Metric by Arya at el., 2019.

    Montonicity tests if adding more positive evidence increases the probability
    of classification in the specified class.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Arya, Vijay, et al. "One explanation does not fit all: A toolkit and taxonomy of ai explainability
        techniques." arXiv preprint arXiv:1909.03012 (2019).
        2) Luss, Ronny, et al. "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).
    """

    @attributes_check
    def __init__(
        self,
        concept_influence: bool = False,
        features_in_step: int = 1,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
        features_in_step (integer): The size of the step, default=1.
        concept_influence (boolean): Indicates whether concept influence metric is used, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
        self.concept_influence = concept_influence
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Arya, Vijay, et al. 'One explanation does not fit all: A toolkit and taxonomy"
                    " of ai explainability techniques.' arXiv preprint arXiv:1909.03012 (2019)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = MonotonicityArya(abs=True, normalise=False)
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
            softmax=softmax,
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

        # Get indices of sorted attributions (ascending).
        a_indices = np.argsort(a)

        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        preds = [None for _ in range(n_perturbations)]

        # Copy the input x but fill with baseline values.
        baseline_value = utils.get_baseline_value(
            value=self.perturb_func_kwargs["perturb_baseline"],
            arr=x,
            return_shape=(1,),
        )
        x_baseline = np.full(x.shape, baseline_value)

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_baseline = self.perturb_func(
                arr=x_baseline,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )

            # Predict on perturbed input x (that was initially filled with a constant 'perturb_baseline' value).
            x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            preds[i_ix] = y_pred_perturb

        return np.all(np.diff(preds) >= 0)

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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch


class MonotonicityNguyen(PerturbationMetric):
    """
    Implementation of Montonicity Metric by Nguyen at el., 2020.

    Monotonicity measures the (Spearman’s) correlation coefficient of the absolute values of the attributions
    and the uncertainty in probability estimation. The paper argues that if attributions are not monotonic
    then they are not providing the correct importance of the feature.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
    """

    @attributes_check
    def __init__(
        self,
        eps: float = 1e-5,
        nr_samples: int = 100,
        features_in_step: int = 1,
        similarity_func: Optional[
            Callable
        ] = None,  # TODO: specify expected function signature
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "uniform",
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
        nr_samples (integer): The number of samples to iterate over, default=100.
        features_in_step (integer): The size of the step, default=1.
        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=correlation_spearman.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
            similarity_func = correlation_spearman
        self.similarity_func = similarity_func

        self.eps = eps
        self.nr_samples = nr_samples
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', threshold value 'eps' and number "
                    "of samples to iterate over 'nr_samples'"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = MonotonicityNguyen(abs=True, normalise=False)
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
            softmax=softmax,
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
        # Predict on input x.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        inv_pred = 1.0 if np.abs(y_pred) < self.eps else 1.0 / np.abs(y_pred)
        inv_pred = inv_pred ** 2

        a = a.flatten()

        # Get indices of sorted attributions (ascending).
        a_indices = np.argsort(a)

        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        atts = [None for _ in range(n_perturbations)]
        vars = [None for _ in range(n_perturbations)]

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]

            y_pred_perturbs = [None for _ in range(self.nr_samples)]

            for sample_id in range(self.nr_samples):

                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    indexed_axes=self.a_axes,
                    **self.perturb_func_kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input)[:, y])
                y_pred_perturbs[sample_id] = y_pred_perturb

            vars[i_ix] = float(
                np.mean((np.array(y_pred_perturbs) - np.array(y_pred)) ** 2) * inv_pred
            )
            atts[i_ix] = float(sum(a[a_ix]))

        return self.similarity_func(a=atts, b=vars)

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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch


class PixelFlipping(PerturbationMetric):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(
        self,
        features_in_step: int = 1,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
        features_in_step (integer): The size of the step, default=1.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=True.
        default_plot_func (callable): Callable that plots the metrics result, if None, the default value is used,
            default=plot_pixel_flipping_experiment.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_pixel_flipping_experiment

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
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                    " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                    "e0130140"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = PixelFlipping(abs=False, normalise=False)
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
            softmax=softmax,
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

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        n_perturbations = len(range(0, len(a_indices), self.features_in_step))
        preds = [None for _ in range(n_perturbations)]
        x_perturbed = x.copy()

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            preds[i_ix] = y_pred_perturb

        return preds

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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [utils.calculate_auc(np.array(i)) for i in self.all_results]


class RegionPerturbation(PerturbationMetric):
    """

    Implementation of Region Perturbation by Samek et al., 2015.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
        neural network has learned." IEEE transactions on neural networks and
        learning systems 28.11 (2016): 2660-2673.

    Current assumptions:
        -Done according to Most Relevant First (MoRF) and Area Over the Perturbation Curve
        (AOPC).
        - 9 x 9 patch sizes was used in the paper as regions, but using 8 x 8
        to make sure non-overlapping
        - they called it "area over the MoRF perturbation curve" it
        looks like a simple deduction of function outputs?

    """

    @attributes_check
    def __init__(
        self,
        patch_size: int = 8,
        order: str = "morf",
        regions_evaluation: int = 100,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "uniform",
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
        patch_size (integer): The patch size for masking, default=8.
        regions_evaluation (integer): The number of regions to evaluate, default=100.
        order (string): Indicates whether attributions are ordered randomly ("random"),
            according to the most relevant first ("morf"), or least relevant first ("lerf"), default="morf".
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=True.
        default_plot_func (callable): Callable that plots the metrics result. If None, the default is used,
            default = plot_region_perturbation_experiment
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_region_perturbation_experiment

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
        self.patch_size = patch_size
        self.order = order.lower()
        self.regions_evaluation = regions_evaluation

        # Asserts and warnings.
        asserts.assert_attributions_order(order=self.order)
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline'"
                    ", the patch size for masking 'patch_size'"
                    " and number of regions to evaluate 'regions_evaluation'"
                ),
                citation=(
                    "Samek, Wojciech, et al. 'Evaluating the visualization of what a deep"
                    " neural network has learned.' IEEE transactions on neural networks and"
                    " learning systems 28.11 (2016): 2660-2673"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = RegionPerturbation(abs=False, normalise=False)
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
            softmax=softmax,
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
        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        patches = []
        x_perturbed = x.copy()

        # Pad input and attributions. This is needed to allow for any patch_size.
        pad_width = self.patch_size - 1
        x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=self.a_axes)
        a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=self.a_axes)

        # Create patches across whole input shape and aggregate attributions.
        att_sums = []
        axis_iterators = [
            range(pad_width, x_pad.shape[axis] - pad_width) for axis in self.a_axes
        ]
        for top_left_coords in itertools.product(*axis_iterators):
            # Create slice for patch.
            patch_slice = utils.create_patch_slice(
                patch_size=self.patch_size,
                coords=top_left_coords,
            )

            # Sum attributions for patch.
            att_sums.append(
                a_pad[utils.expand_indices(a_pad, patch_slice, self.a_axes)].sum()
            )
            patches.append(patch_slice)

        if self.order == "random":
            # Order attributions randomly.
            order = np.arange(len(patches))
            np.random.shuffle(order)

        elif self.order == "morf":
            # Order attributions according to the most relevant first.
            order = np.argsort(att_sums)[::-1]

        elif self.order == "lerf":
            # Order attributions according to the least relevant first.
            order = np.argsort(att_sums)

        else:
            raise ValueError(
                "order must be in ['random', 'morf', 'lerf'] but is: {self.order}"
            )

        # Create ordered list of patches.
        ordered_patches = [patches[p] for p in order]

        # Remove overlapping patches
        blocked_mask = np.zeros(x_pad.shape, dtype=bool)
        ordered_patches_no_overlap = []
        for patch_slice in ordered_patches:
            patch_mask = np.zeros(x_pad.shape, dtype=bool)
            patch_mask[
                utils.expand_indices(patch_mask, patch_slice, self.a_axes)
            ] = True
            # patch_mask_exp = utils.expand_indices(patch_mask, patch_slice, self.a_axes)
            # patch_mask[patch_mask_exp] = True
            intersected = blocked_mask & patch_mask

            if not intersected.any():
                ordered_patches_no_overlap.append(patch_slice)
                blocked_mask = blocked_mask | patch_mask

            if len(ordered_patches_no_overlap) >= self.regions_evaluation:
                break

        # Increasingly perturb the input and store the decrease in function value.
        sub_results = [None for _ in range(len(ordered_patches_no_overlap))]
        for patch_id, patch_slice in enumerate(ordered_patches_no_overlap):
            # Pad x_perturbed. The mode should probably depend on the used perturb_func?
            x_perturbed_pad = utils._pad_array(
                x_perturbed, pad_width, mode="edge", padded_axes=self.a_axes
            )

            # Perturb.
            x_perturbed_pad = self.perturb_func(
                arr=x_perturbed_pad,
                indices=patch_slice,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )

            # Remove Padding
            x_perturbed = utils._unpad_array(
                x_perturbed_pad, pad_width, padded_axes=self.a_axes
            )

            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x and store the difference from predicting on unperturbed input.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            sub_results[patch_id] = y_pred - y_pred_perturb

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
        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # This metric returns a dict of lists, not a list of lists.
        self.last_results = {k: self.last_results[k] for k in range(len(x_batch))}

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [
            utils.calculate_auc(np.array(i))
            for results in self.all_results
            for _, i in results.items()
        ]


class Selectivity(PerturbationMetric):
    """
    Implementation of Selectivity test by Montavon et al., 2018.

    At each iteration, a patch of size 4 x 4 corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """

    @attributes_check
    def __init__(
        self,
        patch_size: int = 8,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
        patch_size (integer): The patch size for masking, default=8.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=True.
        default_plot_func (callable): Callable that plots the metrics result. If None, the default is used,
            default=plot_selectivity_experiment
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_selectivity_experiment

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
        self.patch_size = patch_size

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the patch size for masking"
                    " 'patch_size'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73 (2018): 1-15"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = Selectivity(abs=False, normalise=False)
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
            softmax=softmax,
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
        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        patches = []
        x_perturbed = x.copy()

        # Pad input and attributions. This is needed to allow for any patch_size.
        pad_width = self.patch_size - 1
        x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=self.a_axes)
        a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=self.a_axes)

        # Get patch indices of sorted attributions (descending).
        att_sums = []
        axis_iterators = [
            range(pad_width, x_pad.shape[axis] - pad_width) for axis in self.a_axes
        ]
        for top_left_coords in itertools.product(*axis_iterators):
            # Create slice for patch.
            patch_slice = utils.create_patch_slice(
                patch_size=self.patch_size,
                coords=top_left_coords,
            )

            # Sum attributions for patch.
            att_sums.append(
                a_pad[utils.expand_indices(a_pad, patch_slice, self.a_axes)].sum()
            )
            patches.append(patch_slice)

        # Create ordered list of patches.
        ordered_patches = [patches[p] for p in np.argsort(att_sums)[::-1]]

        # Remove overlapping patches.
        blocked_mask = np.zeros(x_pad.shape, dtype=bool)
        ordered_patches_no_overlap = []
        for patch_slice in ordered_patches:
            patch_mask = np.zeros(x_pad.shape, dtype=bool)
            patch_mask_exp = utils.expand_indices(patch_mask, patch_slice, self.a_axes)
            patch_mask[patch_mask_exp] = True
            intersected = blocked_mask & patch_mask

            if not intersected.any():
                ordered_patches_no_overlap.append(patch_slice)
                blocked_mask = blocked_mask | patch_mask

        # Increasingly perturb the input and store the decrease in function value.
        sub_results = [None for _ in range(len(ordered_patches_no_overlap))]
        for patch_id, patch_slice in enumerate(ordered_patches_no_overlap):

            # Pad x_perturbed. The mode should depend on the used perturb_func.
            x_perturbed_pad = utils._pad_array(
                x_perturbed, pad_width, mode="edge", padded_axes=self.a_axes
            )

            # Perturb.
            x_perturbed_pad = self.perturb_func(
                arr=x_perturbed_pad,
                indices=patch_slice,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )

            # Remove padding.
            x_perturbed = utils._unpad_array(
                x_perturbed_pad, pad_width, padded_axes=self.a_axes
            )

            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x and store the difference from predicting on unperturbed input.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            sub_results[patch_id] = y_pred_perturb

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

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # This metric returns a dict of lists, not a list of lists.
        self.last_results = {k: self.last_results[k] for k in range(len(x_batch))}

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [
            utils.calculate_auc(np.array(i))
            for results in self.all_results
            for i in results.values()
        ]


class SensitivityN(PerturbationMetric):
    """
    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is reported. Sampling is performed using a uniform probability distribution over the features.

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         - Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.
         - I don't get why they have so high correlation in the paper, maybe using a better baseline_value?
         - Also I don't get why correlation is only reported positive?

    """

    @attributes_check
    def __init__(
        self,
        similarity_func: Optional[
            Callable
        ] = None,  # TODO: specify expected function signature
        n_max_percentage: float = 0.8,
        features_in_step: int = 1,
        return_aggregate: bool = True,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "uniform",
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
        similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_pearson.
        n_max_percentage (float): The percentage of features to iteratively evaluatede, fault=0.8.
        features_in_step (integer): The size of the step, default=1.
        return_aggregate (boolean): Indicates whether an aggregated(mean) metric is returned, default=True.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_sensitivity_n_experiment

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
            similarity_func = correlation_pearson
        self.similarity_func = similarity_func

        self.n_max_percentage = n_max_percentage
        self.features_in_step = features_in_step
        self.return_aggregate = return_aggregate

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the patch size for masking "
                    "'patch_size', similarity function 'similarity_func' and the number "
                    "of features to iteratively evaluate 'n_max_percentage'"
                ),
                citation=(
                    "Ancona, Marco, et al. 'Towards better understanding of gradient-based "
                    "attribution methods for deep neural networks.' arXiv preprint "
                    "arXiv:1711.06104 (2017)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = SensitivityN(abs=False, normalise=False)
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
            softmax=softmax,
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

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        # Predict on x.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        att_sums = []
        pred_deltas = []
        x_perturbed = x.copy()

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Sum attributions.
            att_sums.append(float(a[a_ix].sum()))

            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            pred_deltas.append(y_pred - y_pred_perturb)

        # Each list-element of self.last_results will be such a dictionary
        # We will unpack that later in custom_postprocess().
        return {"att_sums": att_sums, "pred_deltas": pred_deltas}

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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_features = int(
            self.n_max_percentage * np.prod(x_batch.shape[2:]) // self.features_in_step
        )

        # Get pred_deltas and att_sums from result list
        sub_results_pred_deltas = [r["pred_deltas"] for r in self.last_results]
        sub_results_att_sums = [r["att_sums"] for r in self.last_results]

        # Re-arrange sublists so that they are sorted by n.
        sub_results_pred_deltas_l = {k: [] for k in range(max_features)}
        sub_results_att_sums_l = {k: [] for k in range(max_features)}

        for k in range(max_features):
            for pred_deltas_instance in sub_results_pred_deltas:
                sub_results_pred_deltas_l[k].append(pred_deltas_instance[k])
            for att_sums_instance in sub_results_att_sums:
                sub_results_att_sums_l[k].append(att_sums_instance[k])

        # Measure similarity for each n.
        self.last_results = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(max_features)
        ]

        if self.return_aggregate:
            self.last_results = [np.mean(self.last_results)]


class Infidelity(PerturbationMetric):
    """
    Implementation of Infidelity by Yeh et al., 2019.

    Explanation infidelity represents the expected mean square error
    between 1) a dot product of an attribution and input perturbation and
    2) difference in model output after significant perturbation.

    Assumptions:
    - The original implementation (https://github.com/chihkuanyeh/saliency_evaluation/
    blob/master/infid_sen_utils.py) supports perturbation of Gaussian noise and squared patches.
    In this implementation, we use squared patches as the default option.

    References:
        1) Chih-Kuan Yeh, Cheng-Yu Hsieh, and Arun Sai Suggala.
        "On the (In)fidelity and Sensitivity of Explanations."
        33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
    """

    @attributes_check
    def __init__(
        self,
        loss_func: str = "mse",
        perturb_patch_sizes: List[int] = None,
        features_in_step: int = 1,
        max_steps_per_input: Optional[int] = None,
        n_perturb_samples: int = 10,
        aggregate: bool = True,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "uniform",
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
        loss_func (string): Loss function, default="mse".
        perturb_patch_sizes (list): List of patch sizes to be perturbed. If None, the defaul is used, default=[4].
        features_in_step (integer): The size of the step, default=1.
        max_steps_per_input (integer): The maximum number of steps per input dimension, default=None.
        n_perturb_samples (integer): The number of samples to be perturbed, default=10.
        aggregate (boolean):  Indicates whether to return the mean values or all values, default=True.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform".
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
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
        if isinstance(loss_func, str):
            if loss_func == "mse":
                loss_func = mse
            # TODO: add more loss functions
            else:
                raise ValueError(f"loss_func must be in ['mse'] but is: {loss_func}")
        self.loss_func = loss_func

        if perturb_patch_sizes is None:
            perturb_patch_sizes = [4]
        self.perturb_patch_sizes = perturb_patch_sizes

        self.features_in_step = features_in_step
        self.max_steps_per_input = max_steps_per_input
        self.n_perturb_samples = n_perturb_samples
        self.aggregate = aggregate

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', perturbation function 'perturb_func',"
                    "number of perturbed samples 'n_perturb_samples', the loss function 'loss_func' "
                    "aggregation boolean 'aggregate'"
                ),
                citation=(
                    "Chih-Kuan, Yeh, et al. 'On the (In)fidelity and Sensitivity of Explanations'"
                    "arXiv:1901.09392 (2019)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = Infidelity(abs=False, normalise=False)
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
            softmax=softmax,
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
        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        sub_results = []

        for _ in range(self.n_perturb_samples):

            sub_sub_results = []

            for patch_size in self.perturb_patch_sizes:

                pred_deltas = np.zeros(
                    (int(a.shape[1] / patch_size), int(a.shape[2] / patch_size))
                )
                a_sums = np.zeros(
                    (int(a.shape[1] / patch_size), int(a.shape[2] / patch_size))
                )
                x_perturbed = x.copy()
                pad_width = patch_size - 1

                for i_x, top_left_x in enumerate(range(0, x.shape[1], patch_size)):

                    for i_y, top_left_y in enumerate(range(0, x.shape[2], patch_size)):

                        # Perturb input patch-wise.
                        x_perturbed_pad = utils._pad_array(
                            x_perturbed, pad_width, mode="edge", padded_axes=self.a_axes
                        )
                        patch_slice = utils.create_patch_slice(
                            patch_size=patch_size,
                            coords=[top_left_x, top_left_y],
                        )

                        x_perturbed_pad = self.perturb_func(
                            arr=x_perturbed_pad,
                            indices=patch_slice,
                            indexed_axes=self.a_axes,
                            **self.perturb_func_kwargs,
                        )

                        # Remove padding.
                        x_perturbed = utils._unpad_array(
                            x_perturbed_pad, pad_width, padded_axes=self.a_axes
                        )

                        # Predict on perturbed input x_perturbed.
                        x_input = model.shape_input(
                            x_perturbed, x.shape, channel_first=True
                        )
                        y_pred_perturb = float(model.predict(x_input)[:, y])

                        x_diff = (x - x_perturbed).flatten()
                        a_diff = np.dot(a.flatten(), x_diff)

                        pred_deltas[i_x][i_y] = y_pred - y_pred_perturb
                        a_sums[i_x][i_y] = np.sum(a_diff)

                sub_sub_results.append(
                    self.loss_func(a=pred_deltas.flatten(), b=a_diff.flatten())
                )

            sub_results.append(np.mean(sub_sub_results))

        if self.aggregate:
            return np.mean(sub_results)
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

        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

        return model, x_batch, y_batch, a_batch, s_batch


class ROAD(PerturbationMetric):
    """
    Implementation of ROAD evaluation strategy by Rong et al., 2022.

    The ROAD approach measures the accuracy of the model on the provided test set at each step of an iterative process
    of removing k most important pixels. At each step k most relevant pixels (MoRF order) are replaced with noisy linear
    imputations which removes bias.

    References:
        1) Rong, Leemann, et al. "Evaluating Feature Attribution: An Information-Theoretic Perspective." arXiv preprint
        arXiv:2202.00449 (2022).
    """

    @attributes_check
    def __init__(
        self,
        percentages: Optional[List[float]] = None,
        noise: float = 0.01,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,  # TODO: specify expected function signature
        perturb_baseline: str = "black",
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
        percentages (list): The list of percentages of the image to be removed, default=list(range(1, 100, 2)).
        noise (noise): Noise added, default=0.01.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=noisy_linear_imputation.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=True.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.

        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = noisy_linear_imputation

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["noise"] = noise

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
        if percentages is None:
            percentages = list(range(1, 100, 2))
        self.percentages = percentages

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', perturbation function 'perturb_func',"
                    "percentage of pixels k removed per iteration 'percentage_in_step'"
                ),
                citation=(
                    "Rong, Leemann, et al. 'Evaluating Feature Attribution: An Information-Theoretic Perspective."
                    "arXiv:2202.00449 (2022)"
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
        softmax: bool = False,
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
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.

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
        >> metric = ROAD(abs=False, normalise=False)
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
            softmax=softmax,
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
        # Order indicies.
        ordered_indices = np.argsort(a, axis=None)[::-1]

        results_instance = [None for _ in self.percentages]
        for p_ix, p in enumerate(self.percentages):

            top_k_indices = ordered_indices[: int(self.a_size * p / 100)]

            x_perturbed = self.perturb_func(
                arr=x,
                indices=top_k_indices,
                **self.perturb_func_kwargs,
            )

            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x and store the difference from predicting on unperturbed input.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            class_pred_perturb = np.argmax(model.predict(x_input))

            # Write a boolean into the percentage results
            results_instance[p_ix] = y == class_pred_perturb

        # Return list of booleans for each percentage
        return results_instance

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
        self.a_size = a_batch[0, :, :].size

        return model, x_batch, y_batch, a_batch, s_batch

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        results = np.array(self.last_results)

        self.last_results = {
            str(percentage): np.sum(results[:, p_ix])
            for p_ix, percentage in enumerate(self.percentages)
        }

        # Calculate accuracy for every number of most important pixels removed.
        self.all_results = {
            percentage: self.last_results[percentage] / len(x_batch)
            for percentage in self.last_results.keys()
        }
