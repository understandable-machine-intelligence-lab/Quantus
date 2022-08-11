"""This module contains the collection of localisation metrics to evaluate attribution-based explanations of neural network models."""
import warnings
from typing import Callable, List, Union, Optional

import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import utils
from ..helpers import plotting
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative


class PointingGame(Metric):
    """
    Implementation of the Pointing Game by Zhang et al., 2018.

    The Pointing Game implements a check whether the point of maximal attribution is on target,
    denoted by a binary mask. High scores are desired as it means, that the maximal attributed pixel belongs to
    an object of the specified class.

    References:
        1) Zhang, Jianming, Baral, Sarah Adel, Lin, Zhe, Brandt, Jonathan, Shen, Xiaohui, and Sclaroff, Stan.
           "Top-Down Neural Attention by Excitation Backprop." International Journal of Computer Vision
           (2018) 126:1084-1102.

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
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch' input as well as if "
                    "the attributions are normalised 'normalise' (and 'normalise_func') "
                    "and/ or taking absolute values of such 'abs'"
                ),
                citation=(
                    "Zhang, Jianming, Baral, Sarah Adel, Lin, Zhe, Brandt, Jonathan, Shen, "
                    "Xiaohui, and Sclaroff, Stan. 'Top-Down Neural Attention by Excitation "
                    "Backprop.' International Journal of Computer Vision, 126:1084-1102 (2018)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
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
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = PointingGame(abs=True, normalise=False)
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
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            # Reshape.
            a = a.flatten()
            s = s.squeeze().flatten().astype(bool)

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Find index of max value.
            max_index = np.where(a == np.max(a))[0]

            # Check if maximum of explanation is on target object class.
            if len(max_index) > 1:
                hit = False
                for pixel in max_index:
                    hit = hit or s[pixel]
            else:
                hit = bool(s[max_index])

            self.last_results.append(
                hit
            )  # ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class AttributionLocalisation(Metric):
    """
    Implementation of the Attribution Localization by Kohlbrenner et al., 2020.

    The Attribution Localization implements the ratio of positive attributions within the target to the overall
    attribution. High scores are desired, as it means, that the positively attributed pixels belong to the
    targeted object class.

    References:
        1) Kohlbrenner M., Bauer A., Nakajima S., Binder A., Wojciech S., Lapuschkin S.
           "Towards Best Practice in Explaining Neural Network Decisions with LRP."
           arXiv preprint arXiv:1910.09840v2 (2020).

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
            weighted (boolean): Indicates whether the weighted variant of the inside-total relevance ratio is used,
            default=False
            max_size (float): The maximum ratio for  the size of the bounding box to image, default=1.0.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.weighted = self.kwargs.get("weighted", False)
        self.max_size = self.kwargs.get("max_size", 1.0)
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        asserts.assert_max_size(max_size=self.max_size)
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch', if size of the ground truth "
                    "mask is taking into account 'weighted' as well as if attributions"
                    " are normalised 'normalise' (and 'normalise_func') and/ or taking "
                    "the absolute values of such 'abs'"
                ),
                citation=(
                    "Kohlbrenner M., Bauer A., Nakajima S., Binder A., Wojciech S., Lapuschkin S. "
                    "'Towards Best Practice in Explaining Neural Network Decisions with LRP."
                    "arXiv preprint arXiv:1910.09840v2 (2020)."
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
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
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = AttributionLocalisation(abs=True, normalise=False)
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
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()
            s = s.flatten().astype(bool)

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)
            else:
                a = np.abs(a)
                warn_func.warn_absolutes_applied()

            # Asserts on attributions.
            if np.all((a < 0.0)):
                raise ValueError("Attributions must not all be less than zero.")
            if not np.any(s):
                raise ValueError(
                    "Segmentation mask must have some non-zero values in its array."
                )

            # Compute ratio.
            size_bbox = float(np.sum(s))
            size_data = np.prod(x.shape[1:])
            ratio = size_bbox / size_data

            # Compute inside/outside ratio.
            inside_attribution = np.sum(a[s])
            total_attribution = np.sum(a)
            inside_attribution_ratio = float(inside_attribution / total_attribution)

            if ratio <= self.max_size:
                if inside_attribution_ratio > 1.0:
                    warnings.warn(
                        "Inside explanation is greater than total explanation"
                        f" ({inside_attribution} > {total_attribution})"
                    )
                if not self.weighted:
                    self.last_results.append(inside_attribution_ratio)
                else:
                    self.last_results.append(float(inside_attribution_ratio * ratio))

        if not self.last_results:
            warnings.warn(
                "Data contains no object with a size below max_size: Results are empty."
            )

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class TopKIntersection(Metric):
    """
    Implementation of the top-k intersection by Theiner et al., 2021.

    The TopKIntersection implements the pixel-wise intersection between a ground truth target object mask and
    an "explainer" mask, the binarized version of the explanation. High scores are desired, as the
    overlap between the ground truth object mask and the attribution mask should be maximal.

    References:
        1) Theiner, Jonas, Müller-Budack Eric, and Ewerth, Ralph. "Interpretable Semantic Photo
        Geolocalization." arXiv preprint arXiv:2104.14995 (2021).
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
            concept_influence (boolean): Indicates whether concept influence metric is used, default=False.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.k = self.kwargs.get("k", 1000)
        self.concept_influence = self.kwargs.get("concept_influence", False)
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch', the number of features to "
                    "consider 'k', if size of the ground truth mask is taking into account"
                    " 'concept_influence' as well as if attributions are normalised "
                    "'normalise' (and 'normalise_func') and/ or taking absolute values "
                    "of such 'abs'"
                ),
                citation=(
                    "Theiner, Jonas, Müller-Budack Eric, and Ewerth, Ralph. 'Interpretable "
                    "Semantic Photo Geolocalization.' arXiv preprint arXiv:2104.14995 (2021)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
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
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = TopKIntersection(abs=True, normalise=False)
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
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)
        asserts.assert_value_smaller_than_input_size(
            x=x_batch_s, value=self.k, value_name="k"
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            top_k_binary_mask = np.zeros(a.shape)
            sorted_indices = np.argsort(a, axis=None)
            np.put_along_axis(
                top_k_binary_mask, sorted_indices[-self.k :], 1, axis=None
            )

            s = s.astype(bool)
            top_k_binary_mask = top_k_binary_mask.astype(bool)

            # Top-k intersection.
            tki = 1.0 / self.k * np.sum(np.logical_and(s, top_k_binary_mask))

            # Concept influence (with size of object normalised tki score).
            if self.concept_influence:
                tki = np.prod(s.shape) / np.sum(s) * tki

            self.last_results.append(tki)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class RelevanceRankAccuracy(Metric):
    """
    Implementation of the Relevance Rank Accuracy by Arras et al., 2021.

    The Relevance Rank Accuracy measures the ratio of high intensity relevances within the ground truth mask GT.
    With P_top-k being the set of pixels sorted by there relevance in decreasing order until the k-th pixels,
    the rank accuracy is computed as: rank accuracy = (|P_top-k intersect GT|) / |GT|. High scores are desired,
    as the pixels with the highest positively attributed scores should be within the bounding box of the targeted
    object.

    References:
        1) Arras, Leila, Osman, Ahmed, and Samek, Wojciech. "Ground Truth Evaluation of Neural Network Explanations
        with CLEVR-XAI." arXiv preprint, arXiv:2003.07258v2 (2021)
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch' as well as if the attributions"
                    " are normalised 'normalise' (and 'normalise_func') and/ or taking "
                    "absolute values of such 'abs'"
                ),
                citation=(
                    "Arras, Leila, Osman, Ahmed, and Samek, Wojciech. 'Ground Truth Evaluation "
                    "of Neural Network Explanations with CLEVR-XAI.' arXiv preprint, "
                    "arXiv:2003.07258v2 (2021)."
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
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
            s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = RelevanceRankAccuracy(abs=True, normalise=False)
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
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()
            s = np.where(s.flatten().astype(bool))[0]

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Size of the ground truth mask.
            k = len(s)

            # Sort in descending order.
            a_sorted = np.argsort(a)[-int(k) :]

            # Calculate hits.
            hits = len(np.intersect1d(s, a_sorted))

            if hits != 0:
                rank_accuracy = hits / float(k)
            else:
                rank_accuracy = 0.0

            self.last_results.append(rank_accuracy)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class RelevanceMassAccuracy(Metric):
    """
    Implementation of the Relevance Rank Accuracy by Arras et al., 2021.

    The Relevance Mass Accuracy computes the ratio of positive attributions inside the bounding box to
    the sum of overall positive attributions. High scores are desired, as the pixels with the highest positively
    attributed scores should be within the bounding box of the targeted object.

    References:
        1) Arras, Leila, Osman, Ahmed, and Samek, Wojciech. "Ground Truth Evaluation of Neural Network Explanations
        with CLEVR-XAI." arXiv preprint, arXiv:2003.07258v2 (2021)
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch' as well as if the attributions"
                    " are normalised 'normalise' (and 'normalise_func') and/ or taking "
                    "absolute values of such 'abs'"
                ),
                citation=(
                    "Arras, Leila, Osman, Ahmed, and Samek, Wojciech. 'Ground Truth Evaluation "
                    "of Neural Network Explanations with CLEVR-XAI.' arXiv preprint, "
                    "arXiv:2003.07258v2 (2021)."
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        **kwargs,
    ):
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
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = RelevanceMassAccuracy(abs=True, normalise=False)
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

            # Get explanation function and make asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Reshape.
            s = s.flatten().astype(bool)

            # Compute inside/outside ratio.
            r_within = np.sum(a[s])
            r_total = np.sum(a)

            mass_accuracy = r_within / r_total

            self.last_results.append(mass_accuracy)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class AUC(Metric):
    """
    Implementation of AUC metric by Fawcett et al., 2006.

    AUC is a ranking metric and  compares the ranking between attributions and a given ground-truth mask

    References:
        1) Fawcett, Tom. 'An introduction to ROC analysis' "Pattern Recognition Letters" Vol 27, Issue 8, 2006

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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch' input as well as if "
                    "absolute values 'abs' are taken of the attributions "
                ),
                citation=(
                    "Fawcett, Tom. 'An introduction to ROC analysis' Pattern Recognition Letters"
                    " Vol 27, Issue 8, (2006)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        Implementation of metric logic to evaluate explanations (a_batch) with respect to some input (x_batch), output
        (y_batch), model (model) and optionally segmentation masks (s_batch) as simulated explanation ground-truth.

        Parameters
        ----------
        model: a torch model e.g., torchvision.models that is to-be-explained
        x_batch: a np.ndarray which contains the inputs that are to-be-explained
        y_batch: a np.ndarray which contains the outputs that are to-be-explained
        a_batch: a Union[np.ndarray, None] which contains pre-computed attributions
        s_batch: a Union[np.ndarray, None] which contains segmentation masks that matches the input
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape by default.
            explain_func (callable): Callable generating attributions, default=Callable.
            device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
            default=None.

        Returns: Union[int, float, list, dict] the output of the metric, which may differ depending on implementation.
        -------

        Examples:
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

            # Generate Saliency attributions of the first batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric instance and evaluate (call) explanations together with a model and some data.
            >> metric = AUC(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **params_call}
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

            # Get explanation function and make asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        asserts.assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, s) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            s = s.flatten()
            s = s.astype(bool)
            a = a.flatten()

            fpr, tpr, _ = roc_curve(y_true=s, y_score=a)
            score = auc(x=fpr, y=tpr)

            self.last_results.append(score)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class Focus(Metric):
    """
    Implementation of Focus evaluation strategy by Arias et. al. 2022

    The Focus is computed through mosaics of instances from different classes, and the explanations these generate.
    Each mosaic contains four images: two images belonging to the target class (the specific class the feature
    attribution method is expected to explain) and the other two are chosen randomly from the rest of classes.
    Thus, the Focus estimates the reliability of feature attribution method’s output as the probability of the sampled
    pixels lying on an image of the target class of the mosaic. This is equivalent to the proportion
    of positive relevance lying on those images.
    
    References:
        1) Anna Arias-Duart, Ferran Parés, Dario Garcia-Gasulla, Victor Gimenez-Abalos. "Focus! Rating XAI Methods 
        and Finding Biases" arXiv preprint arXiv:2109.15035 (2022).
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
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_focus
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        self.mosaic_shape = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "no parameter! No parameters means nothing to be sensistive on."
                ),
                citation=(
                    "Arias-Duart, Anna, et al. 'Focus! Rating XAI Methods and Finding Biases.'"
                    "arXiv:2109.15035 (2022)"
                ),
            )

    def __call__(
        self,
        model: Optional[ModelInterface],
        x_batch: Optional[np.array],
        y_batch: Optional[np.array],
        a_batch: Optional[np.array],
        p_batch: List[tuple],
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
        ----------
            model: a torch model e.g., torchvision.models that is subject to explanation. The model has to be input
                   agnostic i.e. allowing to feed-forward inputs of any size thorugh the network.
            x_batch: a np.ndarray which contains the input mosaics that are explained
            y_batch: a np.ndarray which contains the target classes that are explained
            a_batch: an optional np.ndarray which contains pre-computed attributions i.e., explanations
            p_batch: a List[tuple] which contains the positions of the target class within the mosaic. Each tuple
                     contains 0/1 values referring to (top_left, top_right, bottom_left, bottom_right).
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a list of floats with the evaluation outcome of concerned batch

        Examples
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model with adaptive pooling layer (architecture at quantus/helpers/models).
            >> model = LeNetAdaptivePooling(input_shape=(1, 28, 28))
            >> model.load_state_dict(torch.load("assets/mnist", map_location="cpu", pickle_module=pickle))

            # Load MNIST datasets and make loaders.
            >> transformer = transforms.Compose([transforms.ToTensor()])
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', train=False, transform=transformer, download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, pin_memory=True)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Create mosaics (for example 10 mosaics per class)
            >> x_mosaic_batch, _, _, p_mosaic_batch, y_mosaic_batch = mosaic_creation(images=x_batch,
                                                                       labels=y_batch,
                                                                       mosaics_per_class=10,
                                                                       seed=777)

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Focus()
            >> scores = {method: metric(model=model,
                           x_batch=x_mosaic_batch,
                           y_batch=y_mosaic_batch,
                           a_batch=None,
                           p_batch=p_mosaic_batch,
                           **{"explain_func": explain,
                              "method": "GradCAM",
                              "gc_layer": "model._modules.get('conv_2')",
                              "pos_only": True,
                              "interpolate": (2*28, 2*28),
                              "interpolate_mode": "bilinear",
                              "device": device}) for method in ["GradCAM", "IntegratedGradients"]}

            # Plot example!
            >> metric.plot(results=scores)
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
            try:
                assert model is not None
                assert x_batch is not None
                assert y_batch is not None
            except AssertionError:
                raise ValueError(
                    "Focus requires either a_batch (explanation maps) or "
                    "the necessary arguments to compute it for you (model, x_batch & y_batch)."
                )

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
            iterator = enumerate(zip(a_batch, p_batch))
        else:
            iterator = tqdm(enumerate(zip(a_batch, p_batch)), total=len(x_batch_s))

        self.mosaic_shape = a_batch[0].shape

        for sample, (a, p) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            total_positive_relevance = np.sum(a[a > 0], dtype=np.float64)
            target_positive_relevance = 0

            quadrant_functions_list = [
                self.quadrant_top_left,
                self.quadrant_top_right,
                self.quadrant_bottom_left,
                self.quadrant_bottom_right,
            ]

            for quadrant_p, quadrant_func in zip(p, quadrant_functions_list):
                if not bool(quadrant_p):
                    continue
                quadrant_relevance = quadrant_func(a)
                target_positive_relevance += np.sum(
                    quadrant_relevance[quadrant_relevance > 0]
                )

            focus_score = target_positive_relevance / total_positive_relevance

            self.last_results.append(focus_score)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    def quadrant_top_left(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[
            :, : int(self.mosaic_shape[1] / 2), : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_hmap

    def quadrant_top_right(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[
            :, int(self.mosaic_shape[1] / 2) :, : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_hmap

    def quadrant_bottom_left(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[
            :, : int(self.mosaic_shape[1] / 2), int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_hmap

    def quadrant_bottom_right(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[
            :, int(self.mosaic_shape[1] / 2) :, int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_hmap
