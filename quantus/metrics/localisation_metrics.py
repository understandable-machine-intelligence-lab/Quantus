"""This module contains the collection of localisation metrics to evaluate attribution-based explanations of neural network models."""
import warnings
import numpy as np
from typing import Union, List, Dict
from sklearn.metrics import roc_curve, auc
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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Pointing game metric is likely to be sensitive to the choice of ground truth mask i.e., the 's_batch'"
            "input on the model call as well as if attributions are normalised 'normalise' (and 'normalise_func') and/ "
            "or taking absolute values of such 'abs'. Go over and select "
            "each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Zhang, Jianming, Baral, Sarah Adel, Lin, Zhe, "
            "Brandt, Jonathan, Shen, Xiaohui, and Sclaroff, Stan. 'Top-Down Neural Attention by "
            "Excitation Backprop.' International Journal of Computer Vision (2018) 126:1084-1102.\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Reshape segmentation heatmap from 3 channels to 1.
            s = s.mean(axis=0)
            s = s.astype(bool)

            # Find index of max value
            maxindex = np.where(a == np.max(a))

            # Ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

            # Check if maximum of explanation is on target object class.
            if len(maxindex[0]) > 1:
                hit = 0
                for pixel in maxindex:
                    hit = hit or s[pixel[0], pixel[1]]
            else:
                hit = s[maxindex[0][0], maxindex[1][0]]

            self.last_results.append(hit)

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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.weighted = self.kwargs.get("weighted", False)
        self.max_size = self.kwargs.get("max_size", 1.0)
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Attribution localisation metric is likely to be sensitive to the choice of ground truth "
            "mask i.e., the 's_batch' input on the model call, if size of the ground truth mask is taking into "
            "account 'weighted' as well as if attributions are normalised 'normalise' (and 'normalise_func') and/ "
            "or taking absolute values of such 'abs'. Go over and select each hyperparameter of the metric carefully to"
            " avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Kohlbrenner M., Bauer A., Nakajima S., Binder A., "
            "Wojciech S., Lapuschkin S. 'Towards Best Practice in Explaining Neural Network Decisions with LRP.' "
            "arXiv preprint arXiv:1910.09840v2 (2020).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)
        assert_max_size(max_size=self.max_size)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Reshape segmentation heatmap from 3 channels to 1.
            s = s.mean(axis=0)
            s = s.astype(bool)

            # Asserts on attributions.
            assert not np.all(
                (a < 0.0)
            ), "Attributions should not all be less than zero."
            assert np.any(
                s
            ), "Segmentation mask should have some values in its array that is not zero."

            # Filter positive attribution values.
            a[a < 0.0] = 0.0

            # Compute inside/outside ratio.
            inside_attribution = np.sum(a[s])
            total_attribution = np.sum(a)

            size_bbox = float(np.sum(s))
            size_data = float(np.shape(s)[0] * np.shape(s)[1])
            ratio = size_bbox / size_data

            if ratio <= self.max_size:
                if inside_attribution / total_attribution > 1.0:
                    print(
                        "The inside explanation {} greater than total explanation {}".format(
                            inside_attribution, total_attribution
                        )
                    )

                inside_attribution_ratio = inside_attribution / total_attribution

                if not self.weighted:
                    self.last_results.append(inside_attribution_ratio)

                else:
                    weighted_inside_attribution_ratio = inside_attribution_ratio * (
                        size_data / size_bbox
                    )

                    self.last_results.append(weighted_inside_attribution_ratio)

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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.k = self.kwargs.get("k", 1000)
        self.concept_influence = self.kwargs.get("concept_influence", False)
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Top-k intersection metric is likely to be sensitive to the choice of ground truth "
            "mask i.e., the 's_batch' input on the model call, the number of features to consider 'k',"
            "if size of the ground truth mask is taking into account 'concept_influence' as well as "
            "if attributions are normalised 'normalise' (and 'normalise_func') and/ or taking absolute "
            "values of such 'abs'. Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Theiner, Jonas, Müller-Budack Eric, and Ewerth, "
            "Ralph. 'Interpretable Semantic Photo Geolocalization.' arXiv preprint arXiv:2104.14995 (2021).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

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
            top_k_binary_mask.astype(bool)

            # Top-k intersection.
            tki = 1.0 / self.k * np.sum(np.logical_and(s, top_k_binary_mask))

            # Concept influence (with size of object normalised tki score).
            if self.concept_influence:
                tki = (s.shape[1] * s.shape[2]) / np.sum(s) * tki

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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Relevance rank accuracy metric is likely to be sensitive to the choice of ground truth "
            "mask i.e., the 's_batch' input on the model call as well as "
            "if attributions are normalised 'normalise' (and 'normalise_func') and/ or taking absolute "
            "values of such 'abs'. Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Arras, Leila, Osman, Ahmed, and Samek, Wojciech. "
            "'Ground Truth Evaluation of Neural Network Explanations with CLEVR-XAI.' "
            "arXiv preprint, arXiv:2003.07258v2 (2021).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            s = s.astype(bool)

            sorted_indices = np.argsort(a, axis=None)

            # Size of the ground truth mask.
            k = np.sum(s)
            hits = np.take_along_axis(s, sorted_indices[-int(k) :], axis=None)

            rank_accuracy = np.sum(hits) / float(k)

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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Relevance mass accuracy metric is likely to be sensitive to the choice of ground truth "
            "mask i.e., the 's_batch' input on the model call as well as "
            "if attributions are normalised 'normalise' (and 'normalise_func') and/ or taking absolute "
            "values of such 'abs'. Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Arras, Leila, Osman, Ahmed, and Samek, Wojciech. "
            "'Ground Truth Evaluation of Neural Network Explanations with CLEVR-XAI.' "
            "arXiv preprint, arXiv:2003.07258v2 (2021).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        **kwargs
    ):

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

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

            # Filter positive attribution values.
            a[a < 0.0] = 0.0

            s = s.reshape(self.img_size, self.img_size)
            s = s.astype(bool)

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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe AUC metric is likely to be sensitive to the choice of ground truth "
            "mask i.e., the 's_batch' input on the model call as well as if absolute values are taken on the "
            "attributions 'abs'. Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Fawcett, Tom. 'An introduction to ROC analysis' "
            "Pattern Recognition Letters Vol 27, Issue 8, (2006).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

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
