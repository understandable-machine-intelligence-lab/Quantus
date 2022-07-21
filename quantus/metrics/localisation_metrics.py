"""This module contains the collection of localisation metrics to evaluate attribution-based explanations of neural network models."""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import utils
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
    def __init__(
            self,
            weighted: bool = False,
            abs: bool = False,
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
        weighted (boolean): Indicates whether output score is weighted by size of segmentation map, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
        self.weighted = weighted

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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = PointingGame(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        # Return np.nan as result if segmentation map is empty.
        if np.sum(s) == 0:
            # TODO: add warning
            return np.nan

        # Reshape.
        a = a.flatten()
        s = s.flatten().astype(bool)

        # Find indices with max value.
        max_index = np.argwhere(a == np.max(a))

        # Check if maximum of explanation is on target object class.
        hit = np.any(s[max_index])

        if self.weighted and hit:
            hit = 1 - (np.sum(s) / float(np.prod(s.shape)))
        return hit

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        return model, x_batch, y_batch, a_batch, s_batch


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
    def __init__(
            self,
            weighted: bool = False,
            max_size: float = 1.0,
            abs: bool = True,
            normalise: bool = True,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict] = None,
            softmax: bool = False,
            default_plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs,
    ):
        """
        Parameters
        ----------
        weighted (boolean): Indicates whether the weighted variant of the inside-total relevance ratio is used,
            default=False.
        max_size (float): The maximum ratio for  the size of the bounding box to image, default=1.0.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if not abs:
            # TODO: why are we forcing this instead of just raising a warning?
            # TODO: State this behaviour at least in the docstring.
            abs = True
            warn_func.warn_absolutes_applied()

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
        self.weighted = weighted
        self.max_size = max_size

        # Asserts and warnings.
        self.disable_warnings = disable_warnings
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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = AttributionLocalisation(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        if np.sum(s) == 0:
            return np.nan

        a = a.flatten()
        s = s.flatten().astype(bool)

        # Asserts on attributions.
        if np.all((a < 0.0)):
            raise ValueError("Attributions must not all be less than zero.")

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
                return inside_attribution_ratio
            else:
                return float(inside_attribution_ratio * ratio)

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        for a in a_batch:
            if np.all((a < 0.0)):
                raise ValueError("Attributions must not all be less than zero.")

        for s in s_batch:
            if not np.any(s):
                raise ValueError(
                    "Segmentation mask must have some non-zero values in its array."
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
        if not self.last_results:
            warnings.warn(
                "Data contains no object with a size below max_size: Results are empty."
            )


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
    def __init__(
            self,
            k: int = 1000,
            concept_influence: bool = False,
            abs: bool = False,
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
        k (integer): Top k attributions values to use, default=1000.
        concept_influence (boolean): Indicates whether concept influence metric is used, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
        self.k = k
        self.concept_influence = concept_influence

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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = TopKIntersection(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        if np.sum(s) == 0:
            return np.nan

        s = s.astype(bool)

        top_k_binary_mask = np.zeros(a.shape)
        sorted_indices = np.argsort(a, axis=None)
        np.put_along_axis(
            top_k_binary_mask, sorted_indices[-self.k :], 1, axis=None
        )

        top_k_binary_mask = top_k_binary_mask.astype(bool)

        # Top-k intersection.
        tki = 1.0 / self.k * np.sum(np.logical_and(s, top_k_binary_mask))

        # Concept influence (with size of object normalised tki score).
        if self.concept_influence:
            tki = np.prod(s.shape) / np.sum(s) * tki

        return tki

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        asserts.assert_value_smaller_than_input_size(
            x=x_batch, value=self.k, value_name="k"
        )
        return model, x_batch, y_batch, a_batch, s_batch


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
    def __init__(
            self,
            abs: bool = False,
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
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = RelevanceRankAccuracy(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):

        a = a.flatten()
        s = s.flatten().astype(bool)

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

        return rank_accuracy

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        return model, x_batch, y_batch, a_batch, s_batch


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
    def __init__(
            self,
            abs: bool = False,
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
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = RelevanceRankAccuracy(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):

        a = a.flatten()
        s = np.where(s.flatten().astype(bool))[0]

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

        return rank_accuracy

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        return model, x_batch, y_batch, a_batch, s_batch


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
    def __init__(
            self,
            abs: bool = False,
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
        weighted (boolean): Indicates whether output score is weighted by size of segmentation map, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
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
        >> metric = RelevanceMassAccuracy(abs=True, normalise=False)
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        a = a.flatten()
        s = s.flatten().astype(bool)

        # Compute inside/outside ratio.
        r_within = np.sum(a[s])
        r_total = np.sum(a)

        mass_accuracy = r_within / r_total

        return mass_accuracy

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        return model, x_batch, y_batch, a_batch, s_batch


class AUC(Metric):
    """
    Implementation of AUC metric by Fawcett et al., 2006.

    AUC is a ranking metric and  compares the ranking between attributions and a given ground-truth mask

    References:
        1) Fawcett, Tom. 'An introduction to ROC analysis' "Pattern Recognition Letters" Vol 27, Issue 8, 2006

    """

    @attributes_check
    def __init__(
            self,
            abs: bool = False,
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
        weighted (boolean): Indicates whether output score is weighted by size of segmentation map, default=False.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        softmax (boolean): Indicates wheter to use softmax probabilities or logits in model prediction, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
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
            model,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Optional[np.ndarray],
            s_batch: np.array,
            channel_first: Optional[bool] = None,
            explain_func: Optional[Callable] = None,  # Specify function signature
            explain_func_kwargs: Optional[Dict[str, Any]] = None,
            model_predict_kwargs: Optional[Dict[str, Any]] = None,
            device: Optional[str] = None,
            **kwargs,
    ) -> List[float]:
        """
        Implementation of metric logic to evaluate explanations (a_batch) with respect to some input (x_batch), output
        (y_batch), model (model) and optionally segmentation masks (s_batch) as simulated explanation ground-truth.

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
        last_results: Union[int, float, list, dict] the output of the metric, which may differ depending on implementation.

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

        # Generate Saliency attributions of the first batch of the test set.
        >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
        >> a_batch_saliency = a_batch_saliency.cpu().numpy()

        # Initialise the metric instance and evaluate (call) explanations together with a model and some data.
        >> metric = AUC(abs=True, normalise=False)
        >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
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
            perturb_func: Callable = None,
            perturb_func_kwargs: Optional[Dict] = None,
    ):
        a = a.flatten()
        s = s.flatten().astype(bool)

        fpr, tpr, _ = roc_curve(y_true=s, y_score=a)
        score = auc(x=fpr, y=tpr)
        return score

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
    ) -> Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)
        return model, x_batch, y_batch, a_batch, s_batch
