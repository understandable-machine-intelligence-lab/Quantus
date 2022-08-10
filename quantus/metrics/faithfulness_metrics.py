"""This module contains the collection of faithfulness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import warnings
from typing import Callable, Dict, List, Union

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from .base import Metric
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


class FaithfulnessCorrelation(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            nr_runs (integer): the number of runs (for each input and explanation pair), default=100.
            subset_size (integer): the size of subset, default=224.
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_pearson.
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices.
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        self.nr_runs = self.kwargs.get("nr_runs", 100)
        self.subset_size = self.kwargs.get("subset_size", 224)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.softmax = self.kwargs.get("softmax", False)

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = FaithfulnessCorrelation(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_value_smaller_than_input_size(
            x=x_batch_s, value=self.subset_size, value_name="subset_size"
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            pred_deltas = []
            att_sums = []

            # For each test data point, execute a couple of runs.
            for i_ix in range(self.nr_runs):

                # Randomly mask by subset size.
                a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                pred_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions of the random subset.
                att_sums.append(np.sum(a[a_ix]))

            similarity = self.similarity_func(a=att_sums, b=pred_deltas)
            self.last_results.append(similarity)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class FaithfulnessEstimate(Metric):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            nr_runs (integer): the number of runs (for each input and explanation pair), default=100.
            subset_size (integer): the size of subset, default=224.
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_spearman.
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices.
            features_in_step (integer): the size of the step, default=1.
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", False)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = FaithfulnessEstimate(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                pred_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            self.last_results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class IterativeRemovalOfFeatures(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            segmentation_method (string): Image segmentation method:'slic' or 'felzenszwalb', default="slic"
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="mean"
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            return_aggregate (boolean): indicates whether an aggregated(mean) metric is returned, default=True
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.segmentation_method = self.kwargs.get("segmentation_method", "slic")
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "mean")
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = IROF(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        nr_channels = x_batch_s.shape[1]
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Predict on x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

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
                a_ix = np.nonzero((segments == s_ix).flatten())[0]

                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])

                # Normalise the scores to be within range [0, 1].
                preds.append(float(y_pred_perturb / y_pred))

            self.last_results.append(len(preds) - utils.calculate_auc(np.array(preds)))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def get_aggregated_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return [np.mean(results) for results in self.all_results]


class MonotonicityArya(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black"
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            features_in_step (integer): the size of the step, default=1
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[bool]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of bool(s) with the evaluation outcome of concerned batch

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
            >> metric = MonotonicityArya(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            preds = []

            # Copy the input x but fill with baseline values.
            baseline_value = utils.get_baseline_value(
                value=self.kwargs.get("perturb_baseline", "black"),
                arr=x,
                return_shape=(1,),
            )
            x_baseline = np.full(x.shape, baseline_value)

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_baseline = self.perturb_func(
                    arr=x_baseline,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )

                # Predict on perturbed input x (that was initially filled with a constant 'perturb_baseline' value).
                x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                preds.append(y_pred_perturb)

            self.last_results.append(np.all(np.diff(preds) >= 0))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class MonotonicityNguyen(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform"
            eps (float): Attributions threshold, default=1e-5
            nr_samples (integer): the number of samples to iterate over, default=100
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_spearman
            features_in_step (integer): the size of the step, default=1
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.eps = self.kwargs.get("eps", 1e-5)
        self.nr_samples = self.kwargs.get("nr_samples", 100)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = MonotonicityNguyen(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            # Predict on input x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            inv_pred = 1.0 if np.abs(y_pred) < self.eps else 1.0 / np.abs(y_pred)
            inv_pred = inv_pred**2

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            atts = []
            vars = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]

                y_pred_perturbs = []

                for n in range(self.nr_samples):

                    x_perturbed = self.perturb_func(
                        arr=x,
                        indices=a_ix,
                        indexed_axes=a_axes,
                        **self.kwargs,
                    )
                    asserts.assert_perturbation_caused_change(
                        x=x, x_perturbed=x_perturbed
                    )

                    # Predict on perturbed input x.
                    x_input = model.shape_input(
                        x_perturbed, x.shape, channel_first=True
                    )
                    y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                    y_pred_perturbs.append(y_pred_perturb)

                vars.append(
                    float(
                        np.mean((np.array(y_pred_perturbs) - np.array(y_pred)) ** 2)
                        * inv_pred
                    )
                )
                atts.append(float(sum(a[a_ix])))

            self.last_results.append(self.similarity_func(a=atts, b=vars))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class PixelFlipping(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black"
            features_in_step (integer): the size of the step, default=1
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_pixel_flipping_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = PixelFlipping(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            preds = []
            x_perturbed = x.copy()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    arr=x_perturbed,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                preds.append(y_pred_perturb)

            self.last_results.append(preds)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [utils.calculate_auc(np.array(i)) for i in self.all_results]


class RegionPerturbation(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_patch
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform"
            regions_evaluation (integer): the number of regions to evaluate, default=100
            patch_size (integer): the patch size for masking, default=8
            order (string): indicates whether attributions are ordered randomly ("random"),
            according to the most relevant first ("MoRF"), or least relevant first, default="MoRF"
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_region_perturbation_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.regions_evaluation = self.kwargs.get("regions_evaluation", 100)
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.order = self.kwargs.get("order", "MoRF").lower()
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = {}
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dict of pairs of int(s) and list of float(s) with the evaluation outcome of batch

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
            >> metric = RegionPerturbation(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = {k: None for k in range(len(x_batch_s))}

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for sample, (x, y, a) in iterator:
            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            patches = []
            sub_results = []
            x_perturbed = x.copy()

            # Pad input and attributions. This is needed to allow for any patch_size.
            pad_width = self.patch_size - 1
            x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=a_axes)
            a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=a_axes)

            # Create patches across whole input shape and aggregate attributions.
            att_sums = []
            axis_iterators = [
                range(pad_width, x_pad.shape[axis] - pad_width) for axis in a_axes
            ]
            for top_left_coords in itertools.product(*axis_iterators):
                # Create slice for patch.
                patch_slice = utils.create_patch_slice(
                    patch_size=self.patch_size,
                    coords=top_left_coords,
                )

                # Sum attributions for patch.
                att_sums.append(
                    a_pad[utils.expand_indices(a_pad, patch_slice, a_axes)].sum()
                )
                patches.append(patch_slice)

            if self.order == "random":
                # Order attributions randomly.
                order = np.arange(len(patches))
                np.random.shuffle(order)

            elif self.order == "morf":
                # Order attributions according to the most relevant first.
                order = np.argsort(att_sums)[::-1]

            else:
                # Order attributions according to the least relevant first.
                order = np.argsort(att_sums)

            # Create ordered list of patches.
            ordered_patches = [patches[p] for p in order]

            # Remove overlapping patches
            blocked_mask = np.zeros(x_pad.shape, dtype=bool)
            ordered_patches_no_overlap = []
            for patch_slice in ordered_patches:
                patch_mask = np.zeros(x_pad.shape, dtype=bool)
                patch_mask[utils.expand_indices(patch_mask, patch_slice, a_axes)] = True
                intersected = blocked_mask & patch_mask

                if not intersected.any():
                    ordered_patches_no_overlap.append(patch_slice)
                    blocked_mask = blocked_mask | patch_mask

                if len(ordered_patches_no_overlap) >= self.regions_evaluation:
                    break

            # Increasingly perturb the input and store the decrease in function value.
            for patch_slice in ordered_patches_no_overlap:
                # Pad x_perturbed. The mode should probably depend on the used perturb_func?
                x_perturbed_pad = utils._pad_array(
                    x_perturbed, pad_width, mode="edge", padded_axes=a_axes
                )

                # Perturb.
                x_perturbed_pad = self.perturb_func(
                    arr=x_perturbed_pad,
                    indices=patch_slice,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )

                # Remove Padding
                x_perturbed = utils._unpad_array(
                    x_perturbed_pad, pad_width, padded_axes=a_axes
                )

                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])

                sub_results.append(y_pred - y_pred_perturb)

            self.last_results[sample] = sub_results

        if self.return_aggregate:
            print(
                "A 'return_aggregate' functionality is not implemented for this metric."
            )

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [
            utils.calculate_auc(np.array(i))
            for results in self.all_results
            for _, i in results.items()
        ]


class Selectivity(Metric):
    """
    Implementation of Selectivity test by Montavon et al., 2018.

    At each iteration, a patch of size 4 x 4 corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    Note: Plotting only works when return_auc=False.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black"
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            patch_size (integer): the patch size for masking, default=8
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_selectivity_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = {}
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dict of pairs of int(s) and list of float(s) with the evaluation outcome of batch

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
            >> metric = Selectivity(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = {k: None for k in range(len(x_batch_s))}

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for sample, (x, y, a) in iterator:

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            patches = []
            sub_results = []
            x_perturbed = x.copy()

            # Pad input and attributions. This is needed to allow for any patch_size.
            pad_width = self.patch_size - 1
            x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=a_axes)
            a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=a_axes)

            # Get patch indices of sorted attributions (descending).
            att_sums = []
            axis_iterators = [
                range(pad_width, x_pad.shape[axis] - pad_width) for axis in a_axes
            ]
            for top_left_coords in itertools.product(*axis_iterators):
                # Create slice for patch.
                patch_slice = utils.create_patch_slice(
                    patch_size=self.patch_size,
                    coords=top_left_coords,
                )

                # Sum attributions for patch.
                att_sums.append(
                    a_pad[utils.expand_indices(a_pad, patch_slice, a_axes)].sum()
                )
                patches.append(patch_slice)

            # Create ordered list of patches.
            ordered_patches = [patches[p] for p in np.argsort(att_sums)[::-1]]

            # Remove overlapping patches.
            blocked_mask = np.zeros(x_pad.shape, dtype=bool)
            ordered_patches_no_overlap = []
            for patch_slice in ordered_patches:
                patch_mask = np.zeros(x_pad.shape, dtype=bool)
                patch_mask[utils.expand_indices(patch_mask, patch_slice, a_axes)] = True
                intersected = blocked_mask & patch_mask

                if not intersected.any():
                    ordered_patches_no_overlap.append(patch_slice)
                    blocked_mask = blocked_mask | patch_mask

            # Increasingly perturb the input and store the decrease in function value.
            for patch_slice in ordered_patches_no_overlap:

                # Pad x_perturbed. The mode should depend on the used perturb_func.
                x_perturbed_pad = utils._pad_array(
                    x_perturbed, pad_width, mode="edge", padded_axes=a_axes
                )

                # Perturb.
                x_perturbed_pad = self.perturb_func(
                    arr=x_perturbed_pad,
                    indices=patch_slice,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )

                # Remove padding.
                x_perturbed = utils._unpad_array(
                    x_perturbed_pad, pad_width, padded_axes=a_axes
                )

                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])

                sub_results.append(y_pred_perturb)

            self.last_results[sample] = sub_results

        if self.return_aggregate:
            print(
                "A 'return_aggregate' functionality is not implemented for this metric."
            )

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return [
            utils.calculate_auc(np.array(i))
            for results in self.all_results
            for _, i in results.items()
        ]


class SensitivityN(Metric):
    """
    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is reported. Sampling is performed using a uniform probability distribution over the features.

    Note: Plotting only works when return_auc=False.

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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_pearson
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="uniform"
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            n_max_percentage (float): the percentage of features to iteratively evaluatede, default=0.
            features_in_step (integer): the size of the step, default=1
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
            return_aggregate (boolean): indicates whether an aggregated(mean) metric is returned, default=True
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_sensitivity_n_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.n_max_percentage = self.kwargs.get("n_max_percentage", 0.8)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = SensitivityN(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

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
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        max_features = int(
            self.n_max_percentage
            * np.prod(x_batch_s.shape[2:])
            // self.features_in_step
        )

        sub_results_pred_deltas = {k: [] for k in range(len(x_batch_s))}
        sub_results_att_sums = {k: [] for k in range(len(x_batch_s))}

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for sample, (x, y, a) in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on x.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            att_sums = []
            pred_deltas = []
            x_perturbed = x.copy()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    arr=x_perturbed,
                    indices=a_ix,
                    indexed_axes=a_axes,
                    **self.kwargs,
                )
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Sum attributions.
                att_sums.append(float(a[a_ix].sum()))

                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                y_pred_perturb = float(model.predict(x_input, **self.kwargs)[:, y])
                pred_deltas.append(y_pred - y_pred_perturb)

            sub_results_att_sums[sample] = att_sums
            sub_results_pred_deltas[sample] = pred_deltas

        # Re-arrange sublists so that they are sorted by n.
        sub_results_pred_deltas_l = {k: [] for k in range(max_features)}
        sub_results_att_sums_l = {k: [] for k in range(max_features)}

        for k in range(max_features):
            for sublist1 in list(sub_results_pred_deltas.values()):
                sub_results_pred_deltas_l[k].append(sublist1[k])
            for sublist2 in list(sub_results_att_sums.values()):
                sub_results_att_sums_l[k].append(sublist2[k])

        # Measure similarity for each n.
        self.last_results = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(max_features)
        ]

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class Infidelity(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            loss_func (string): Loss function, default="mse"
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black"
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
            perturb_patch_sizes (list): Size of patches to be perturbed, default=[4]
            features_in_step (integer): the size of the step, default=1

            n_perturb_samples (integer): the number of samples to be perturbed, default=10

        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.loss_func = self.kwargs.get("loss_func", mse)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_patch_sizes = self.kwargs.get("perturb_patch_sizes", [4])
        self.n_perturb_samples = self.kwargs.get("n_perturb_samples", 10)

        self.last_results = []
        self.all_results = []

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
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = Infidelity(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)
        nr_channels = x_batch_s.shape[1]

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        if a_batch is None:
            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch_s)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(
                model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
            )

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

                        for i_y, top_left_y in enumerate(
                            range(0, x.shape[2], patch_size)
                        ):

                            # Perturb input patch-wise.
                            x_perturbed_pad = utils._pad_array(
                                x_perturbed, pad_width, mode="edge", padded_axes=a_axes
                            )
                            patch_slice = utils.create_patch_slice(
                                patch_size=patch_size,
                                coords=[top_left_x, top_left_y],
                            )

                            x_perturbed_pad = self.perturb_func(
                                arr=x_perturbed_pad,
                                indices=patch_slice,
                                indexed_axes=a_axes,
                                **self.kwargs,
                            )

                            # Remove padding.
                            x_perturbed = utils._unpad_array(
                                x_perturbed_pad, pad_width, padded_axes=a_axes
                            )

                            # Predict on perturbed input x_perturbed.
                            x_input = model.shape_input(
                                x_perturbed, x.shape, channel_first=True
                            )
                            y_pred_perturb = float(
                                model.predict(
                                    x_input, softmax_act=False, **self.kwargs
                                )[:, y]
                            )

                            x_diff = x - x_perturbed
                            a_diff = np.dot(
                                np.repeat(a, repeats=nr_channels, axis=0), x_diff
                            )

                            pred_deltas[i_x][i_y] = y_pred - y_pred_perturb
                            a_sums[i_x][i_y] = np.sum(a_diff)

                    sub_sub_results.append(
                        self.loss_func(a=pred_deltas.flatten(), b=a_sums.flatten())
                    )

                sub_results.append(np.mean(sub_sub_results))

            if self.return_aggregate:
                self.last_results.append(self.aggregate_func(sub_results))
            else:
                self.last_results.append(sub_results)

        self.all_results.append(self.last_results)

        return self.last_results


class ROAD(Metric):
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
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices.
            percentages (list): the list of percentages of the image to be removed, default=list(range(1, 100, 2)).
            noise (noise): noise added, default=0.01.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_region_perturbation_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get("perturb_func", noisy_linear_imputation)
        self.percentages = self.kwargs.get("percentages", list(range(1, 100, 2)))
        self.noise = self.kwargs.get("noise", 0.01)
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', perturbation function 'perturb_func',"
                    "percentage of pixels k removed per iteration 'percentage_in_step'"
                ),
                citation=(
                    "Rong, Leemann, et al. 'Evaluating Feature Attribution: An Information-Theoretic Perspective.' "
                    "arXiv:2202.00449 (2022)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            all_results: a dictionary whose values contains a list of float(s) with the evaluation for every percentage
            of pixels removed.

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
            >> metric = ROAD(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),  # *len(self.percentages),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        self.last_results = []
        self.all_results = []
        accuraies = {p: [] for p in self.percentages}

        for sample, (x, y, a) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Order indicies.
            ordered_indices = np.argsort(a, axis=None)[::-1]

            for p in self.percentages:

                top_k_indices = ordered_indices[: int(x.size * (p / 100))]

                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=top_k_indices,
                    **self.kwargs,
                )

                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)

                y_pred_label = np.argmax(
                    model.predict(x_input, softmax_act=True, **self.kwargs)
                )

                # Calculate accuracy for every number of most important pixels removed.
                accuraies[p].append(float(y == y_pred_label))

        for p in self.percentages:
            self.last_results.append(np.mean(accuraies[p]))

        if self.return_aggregate:
            print(
                "A 'return_aggregate' functionality is not implemented for this metric."
            )

        self.all_results.append(self.last_results)

        return self.last_results


class Sufficiency(Metric):
    """

    The (global) sufficiency metric measures the expected local sufficiency. Local sufficiency measures the probability
    of the prediction label for a given datapoint coinciding with the prediction labels of other data points that the
    same explanation applies to. For example, if the explanation of a given image is "contains zebra", the local
    sufficiency metric measures the probability a different that contains zebra having the same prediction label.

    References:
         1) Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. "Framework for Evaluating Faithfulness of Local
            Explanations." arXiv preprint arXiv:2202.00734 (2022).

    Assumptions:
        - We assume that a given explanation applies to anothers data point if the distance between this explanation
        and the explanations of the data point is under the user-defined threshold.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            threshold (float): Distance threshold, default=0.6
            distance_func (string): Distance function, default = "seuclidean". ( see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html for more)
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.threshold = self.kwargs.get("threshold", 0.6)
        self.distance_func = self.kwargs.get("distance_func", "seuclidean")

        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "distance threshold that determines if images share an attribute 'threshold', "
                    "distance function 'distance_func'"
                ),
                citation=(
                    "Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. 'Framework for Evaluating Faithfulness of "
                    "Explanations.' arXiv preprint arXiv:2202.00734 (2022)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            metric: value of the metric.

        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        asserts.assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        a_batch_flat = a_batch.reshape(a_batch.shape[0], -1)
        dist_matrix = cdist(a_batch_flat, a_batch_flat, self.distance_func, V=None)
        dist_matrix = self.normalise_func(dist_matrix)
        a_sim_matrix = np.zeros_like(dist_matrix)
        a_sim_matrix[dist_matrix <= self.threshold] = 1

        # Predict on input.
        x_input = model.shape_input(
            x_batch, x_batch[0].shape, channel_first=True, batch=True
        )
        y_pred_classes = np.argmax(
            model.predict(x_input, softmax_act=True, **self.kwargs), axis=1
        ).flatten()

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, a_sim_matrix))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, a_sim_matrix)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, a_sim) in iterator:

            pred_a = y_pred_classes[ix]
            low_dist_a = np.argwhere(a_sim == 1.0).flatten()
            low_dist_a = low_dist_a[low_dist_a != ix]
            pred_low_dist_a = y_pred_classes[low_dist_a]

            if len(low_dist_a) == 0:
                self.last_results.append(0)
            else:
                self.last_results.append(
                    np.sum(pred_low_dist_a == pred_a) / len(low_dist_a)
                )

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results
