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
from ..helpers.normalize_func import *


class PointingGame(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the Pointing Game from Zhang et al. 2018,
    that implements the check if the maximal attribution is on target.

    High scores are desired as it means, that the maximal attributed pixel belongs to an object of the specified class.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
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

        # TODO. assert is binary mask for s_batch

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            # Reshape segmentation heatmap from 3 channels to 1.
            s = s.mean(axis=0)

            # Find index of max value
            maxindex = np.where(a == np.max(a))

            # ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

            # check if maximum of explanation is on target object class
            # case max is at more than one pixel
            if len(maxindex[0]) > 1:
                hit = 0
                for pixel in maxindex:
                    hit = hit or s[pixel[0], pixel[1]]
            else:
                hit = s[maxindex[0][0], maxindex[1][0]]

            self.last_results.append(hit)

        self.all_results.append(self.last_results)

        return self.last_results


class AttributionLocalization(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the attribution localization from Kohlbrenner et al. 2020,
    that implements the ratio of attribution within target to the overall attribution.

    High scores are desired, as it means, that the positively attributed pixels belong to the targeted object class.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.weighted = self.kwargs.get("weighted", False)
        self.max_size = self.kwargs.get("max_size", 1.0)
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

        # Asserts.
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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
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

        # TODO. assert is binary mask for s_batch

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            # Reshape segmentation heatmap from 3 channels to 1.
            s = s.mean(axis=0)
            s = s.astype(bool)

            # Asserts on attributions.
            assert not np.all((a < 0.0)), "Attributions should not all be less than zero."
            assert np.any(s), "Segmentation mask should have some values in its array that is not zero."

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
            warnings.warn("Data contains no object with a size below max_size: Results are empty.")

        self.all_results.append(self.last_results)

        return self.last_results


class TopKIntersection(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the top-k intersection from Theiner et al. 2021,
    that implements the pixel-wise intersection between ground truth and an "explainer" mask.

    High scores are desired, as the overlap between the ground truth object mask
    and the attribution mask should be maximal.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.k = self.kwargs.get("k", 1000)
        self.concept_influence = self.kwargs.get("concept_influence", False)
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
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

        # TODO. assert is binary mask for s_batch

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            top_k_binary_mask = np.zeros(a.shape)

            # TODO. e.g. for sign independent xai methods take the abs of the attribution before ordering the indices
            sorted_indices = np.argsort(a, axis=None)
            np.put_along_axis(top_k_binary_mask, sorted_indices[-self.k:], 1, axis=None)

            s = s.astype(bool)
            top_k_binary_mask.astype(bool)

            # Top-k intersection.
            tki = 1./self.k * np.sum(np.logical_and(s, top_k_binary_mask))

            # Concept influence (with size of object normalized tki score).
            if self.concept_influence:
                tki = (s.shape[1] * s.shape[2]) / np.sum(s) * tki

            self.last_results.append(tki)

        self.all_results.append(self.last_results)

        return self.last_results


class RelevanceRankAccuracy(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the relevance rank accuracy from Arras et al. 2021,
    that measures the ratio of high intensity relevances within the ground truth mask.

    High scores are desired, as the pixels with the highest positively attributed scores
    should be within the bounding box of the targeted object.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
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

        # TODO. assert is binary mask for s_batch

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            k = np.sum(s)   # size of ground truth mask

            # TODO. e.g. for sign independent xai methods take the abs of the attribution before ordering the indices
            sorted_indices = np.argsort(a, axis=None)
            hits = np.take_along_axis(s, sorted_indices[-int(k):], axis=None)

            rank_accuracy = np.sum(hits)/float(k)

            self.last_results.append(rank_accuracy)

        self.all_results.append(self.last_results)

        return self.last_results


class AUC(Metric):
    """
    TODO. Rewrite docstring.

    AUC metric.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
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

        # TODO. assert is binary mask for s_batch

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            s = s.flatten()
            a = a.flatten()

            fpr, tpr, _ = roc_curve(y_true=s, y_score=a)
            score = auc(x=fpr, y=tpr)

            self.last_results.append(score)

        self.all_results.append(self.last_results)

        return self.last_results


class RelevanceMassAccuracy(Metric):
    """
    TODO. Rewrite docstring.
    Implementation of the relevance mass accuracy from Arras et al. 2021,
    that computes the ratio of relevance inside the bounding box to the sum of the overall relevance.
    High scores are desired, as the pixels with the highest positively attributed scores
    should be within the bounding box of the targeted object.
    Current assumptions:
    s_batch is binary and shapes are equal
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.last_results = []
        self.all_results = []

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
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = kwargs.get("explain_func", Callable)
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

        # TODO. Assert is binary mask for s_batch.

        for sample, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            # Asserts on attributions.
            assert not np.all((a < 0.0)), "Attributions should not all be less than zero."
            assert np.any(s), "Segmentation mask should have some values in its array that is not zero."

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