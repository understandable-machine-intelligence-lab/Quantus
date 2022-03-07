"""This module contains the collection of localisation metrics to evaluate attribution-based explanations of neural network models."""
from typing import Union, List, Dict
import warnings

import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from .base import Metric
from ..helpers.utils import *
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *
from ..helpers.normalise_func import *
from ..helpers.warn_func import *


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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            # Reshape.
            a = a.flatten()
            s = s.squeeze().flatten().astype(bool)

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Find index of max value.
            max_index = np.where(a == np.max(a))[0]

            # Check if maximum of explanation is on target object class.
            if len(max_index) > 1:
                hit = False
                for pixel in max_index:
                    hit = hit or s[pixel]
            else:
                hit = bool(s[max_index])

            self.last_results.append(hit)

            # Ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

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
            weighted (boolean): Indicates whether the weighted variant of the inside-total relevance ratio is used,
            default=False.
            max_size (float): The maximum ratio for  the size of the bounding box to image, default=1.0.
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        assert_max_size(max_size=self.max_size)
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:
            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()
            s = s.flatten().astype(bool)

            if self.abs:
                a = np.abs(a)
            else:
                a = np.abs(a)
                print(
                    "An absolute operation is applied on the attributions (regardless if set 'abs' parameter is False)"
                    "since inconsistent results can be expected otherwise."
                )

            if self.normalise:
                a = self.normalise_func(a)

            # Asserts on attributions.
            assert not np.all(
                (a < 0.0)
            ), "Attributions should not all be less than zero."
            assert np.any(
                s
            ), "Segmentation mask should have some values in its array that is not zero."

            # Compute ratio.
            size_bbox = float(np.sum(s))
            size_data = float(self.img_size * self.img_size)
            ratio = size_bbox / size_data

            # Compute inside/outside ratio.
            inside_attribution = np.sum(a[s])
            total_attribution = np.sum(a)
            inside_attribution_ratio = float(inside_attribution / total_attribution)

            if ratio <= self.max_size:
                if inside_attribution_ratio > 1.0:
                    print(
                        "The inside explanation {} greater than total explanation {}".format(
                            inside_attribution, total_attribution
                        )
                    )
                if not self.weighted:
                    self.last_results.append(inside_attribution_ratio)
                else:
                    self.last_results.append(float(inside_attribution_ratio * ratio))

        if not self.last_results:
            warnings.warn(
                "Data contains no object with a size below max_size: Results are empty."
            )

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
            concept_influence (boolean): Indicates whether concept influence metric is used, default=False.
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

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
                tki = (self.img_size * self.img_size) / np.sum(s) * tki

            self.last_results.append(tki)

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()
            s = np.where(s.flatten().astype(bool))[0]

            if self.abs:
                a = np.abs(a)
            else:
                a = np.abs(a)
                print(
                    "An absolute operation is applied on the attributions (regardless if set 'abs' parameter is False)"
                    "since inconsistent results can be expected otherwise."
                )

            if self.normalise:
                a = self.normalise_func(a)

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Get explanation function and make asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Asserts on attributions.
            assert not np.all(
                (a < 0.0)
            ), "Attributions should not all be less than zero."
            assert np.any(
                s
            ), "Segmentation mask should have some values in its array that is not zero."

            # Reshape.
            s = s.flatten().astype(bool)

            # Compute inside/outside ratio.
            r_within = np.sum(a[s])
            r_total = np.sum(a)

            mass_accuracy = r_within / r_total

            self.last_results.append(mass_accuracy)

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
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
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
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
            nr_channels (integer): Number of images, default=second dimension of the input.
            img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Get explanation function and make asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch_s, s_batch=s_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, s_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, s_batch)),
                total=len(x_batch_s),
            )

        for ix, (x, y, a, s) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            s = s.flatten()
            s = s.astype(bool)
            a = a.flatten()

            fpr, tpr, _ = roc_curve(y_true=s, y_score=a)
            score = auc(x=fpr, y=tpr)

            self.last_results.append(score)

        self.all_results.append(self.last_results)

        return self.last_results
