import numpy as np
from typing import Union
import warnings

from .base import Metric
from ..helpers.explanation_func import *


class LocalizationTest(Metric):
    """ Implements basis functionality for the following evaluation measures:

        • Pointing Game (Zhang et al., 2018)
        • Attribution Localization (Kohlbrenner et al., 2020)
        • TKI (Theiner et al., 2021)
        • Relevance Rank Accuracy (Arras et al., 2021)
        • Relevance Mass Accuracy (Arras et al., 2021)
     """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.perturb_func = self.kwargs.get("perturb_func", None)

        self.agg_function = kwargs.get("agg_function", np.mean)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        s_batch: np.array,
        **kwargs
    ):

        raise NotImplementedError("Abstract Class, please specify a Localization Metric.")

    def check_assertions(self,
                         model,
                         x_batch: np.array,
                         y_batch: Union[np.array, int],
                         a_batch: Union[np.array, None],
                         s_batch: np.array,
                         **kwargs
                         ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        if not isinstance(y_batch, int):
            assert (
                    np.shape(x_batch)[0] == np.shape(y_batch)[0]
            ), "Target should by an Integer or a list with the same number of samples as the data."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."
        assert (
                np.shape(x_batch)[1] == np.shape(a_batch)[1]
        ), "Data and attributions should have a corresponding shape."
        assert (
                np.shape(x_batch)[0] == np.shape(s_batch)[0]
        ), "Inputs and segmentation masks should include the same number of samples."
        assert (
                np.shape(a_batch) == np.shape(s_batch)
        ), "Attributions and segmentation masks should have the same shape."

        return True


class PointingGame(LocalizationTest):
    """
    Implementation of the Pointing Game from Zhang et al. 2018,
    that implements the check if the maximal attribution is on target.

    High scores are desired as it means, that the maximal attributed pixel belongs to an object of the specified class.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs
        super(PointingGame, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            s_batch: np.array,
            **kwargs
    ):

        if a_batch is None:
            a_batch = explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
            a_batch = a_batch.cpu().numpy()
        self.check_assertions(model, x_batch, y_batch, a_batch, s_batch, **kwargs)

        # ToDo: assert is binary mask for s_batch

        results = []
        for ix, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):
            # find index of max value
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

            results.append(hit)

        return results


class AttributionLocalization(LocalizationTest):
    """
    Implementation of the attribution localization from Kohlbrenner et al. 2020,
    (also Relevance Mass Accuracy by Arras et al. 2021)
    that implements the ratio of attribution within target to the overall attribution.

    High scores are desired, as it means, that the positively attributed pixels belong to the targeted object class.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs
        self.weighted = self.kwargs.get("weighted", False)
        self.max_size = self.kwargs.get("max_size", 1.0)

        super(AttributionLocalization, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            s_batch: np.array,
            **kwargs
    ):
        assert (
            (self.max_size > 0.) and (self.max_size <= 1.)
        ), "max_size must be between 0. and 1."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
        self.check_assertions(model, x_batch, y_batch, a_batch, s_batch, **kwargs)

        # ToDo: assert is binary mask for s_batch

        results = []
        for ix, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            assert not np.all((a < 0.0))
            assert np.any(s)

            # filter positive attribution values
            a[a < 0.0] = 0.0

            s = s.astype(bool)

            # compute inside/outside ratio
            inside_attribution = np.sum(a[s])
            total_attribution = np.sum(a)

            size_bbox = float(np.sum(s))
            size_data = float(np.shape(s)[0] * np.shape(s)[1])
            ratio = size_bbox / size_data

            if ratio <= self.max_size:
                if inside_attribution / total_attribution > 1.0:
                    print(
                        "inside explanation {} greater than total explanation {}".format(
                            inside_attribution, total_attribution
                        )
                    )

                inside_attribution_ratio = inside_attribution / total_attribution

                if not self.weighted:
                    results.append(inside_attribution_ratio)

                else:
                    weighted_inside_attribution_ratio = inside_attribution_ratio * (
                            size_data / size_bbox
                    )

                    results.append(weighted_inside_attribution_ratio)

        if not results:
            warnings.warn("Data contains no object with a size below max_size: Results are empty.")

        return results


class TopKIntersection(LocalizationTest):
    """
    Implementation of the top-k intersection from Theiner et al. 2021,
    that implements the pixel-wise intersection between ground truth and an "explainer" mask.

    High scores are desired, as the overlap between the ground truth object mask
    and the attribution mask should be maximal.

    Current assumptions:
    s_batch is binary and shapes are equal
    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs
        self.k = self.kwargs.get("k", 1000)

        super(TopKIntersection, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            s_batch: np.array,
            **kwargs
    ):

        if a_batch is None:
            a_batch = explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
            a_batch = a_batch.cpu().numpy()
        self.check_assertions(model, x_batch, y_batch, a_batch, s_batch, **kwargs)

        # ToDo: assert is binary mask for s_batch

        results = []
        for ix, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            top_k_binary_mask = np.zeros(a.shape)
            # ToDo: e.g. for sign independent xai methods take the abs of the attribution before ordering the indices
            sorted_indices = np.argsort(a, axis=None)
            np.put_along_axis(top_k_binary_mask, sorted_indices[-self.k:], 1, axis=None)

            s = s.astype(bool)
            top_k_binary_mask.astype(bool)

            # top-k intersection
            tki = 1./self.k * np.sum(np.logical_and(s, top_k_binary_mask))

            # concept influence (with size of object normalized tki score)
            ci = (s.shape[0]*s.shape[1])/np.sum(s) * tki    # ToDo: incorporate into results

            results.append(tki)

        return results


class RelevanceRankAccuracy(LocalizationTest):
    """
    Implementation of the relevance rank accuracy from Arras et al. 2021,
    that measures the ratio of high intensity relevances within the ground truth mask.

    High scores are desired, as the pixels with the highest positively attributed scores
    should be within the bounding box of the targeted object.

    Current assumptions:
    s_batch is binary and shapes are equal
    """
    def __init__(self):

        super(RelevanceRankAccuracy, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            s_batch: np.array,
            **kwargs
    ):

        if a_batch is None:
            a_batch = explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
            a_batch = a_batch.cpu().numpy()
        self.check_assertions(model, x_batch, y_batch, a_batch, s_batch, **kwargs)

        # ToDo: assert is binary mask for s_batch

        results = []
        for ix, (x, y, a, s) in enumerate(zip(x_batch, y_batch, a_batch, s_batch)):

            k = np.sum(s)   # size of ground truth mask

            # ToDo: e.g. for sign independent xai methods take the abs of the attribution before ordering the indices
            sorted_indices = np.argsort(a, axis=None)
            hits = np.take_along_axis(s, sorted_indices[-k:], axis=None)

            rank_accuracy = np.sum(hits)/float(k)

            results.append(rank_accuracy)

        return results
