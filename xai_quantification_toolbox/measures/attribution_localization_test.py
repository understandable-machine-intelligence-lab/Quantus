import numpy as np
from typing import Union

from .base import Measure


class LocalizationTest(Measure):
    """ Implements basis for all Localization Test Measures. """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.perturbation_function = params.get("perturbation_function", None)
        self.localization_function = params.get(
            "localization_function", "pointing_game"
        )
        self.agg_function = kwargs.get("agg_function", np.mean)

        super(LocalizationTest, self).__init__()


    def __call__(
        self,
        model,
        inputs: np.array,
        targets: Union[np.array, int, None],
        attributions: Union[np.array, None],
        **kwargs
    ):

        results = []

        for s, sample in enumerate(inputs):

            sub_results = []

            for target in targets:

                binary_mask = get_binary_mask(sample, targets[s])

            if self.perturbation_function:
                attribution = self.perturbation_function(attributions[s])
            else:
                attribution = attributions[s]

                sub_results.append(
                    self.localization_function(sample, attribution, binary_mask)
                )

            results.append(self.agg_function(sub_results))

        return results


    @staticmethod
    def pointing_game(attribution, binary_mask):
        """Implements the Pointing Game as described in
        Zhang et. al., 2018, Top-Down Neural Attention by Exitation Backprop
        """

        assert binary_mask.shape == attribution.shape

        # find index of max value
        maxindex = np.where(attribution == np.max(attribution))

        # ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

        # check if maximum of explanation is on target object class
        # case max is at more than one pixel
        if len(maxindex[0]) > 1:
            hit = 0
            for pixel in maxindex:
                hit = hit or binary_mask[pixel[0], pixel[1]]
        else:
            hit = binary_mask[maxindex[0], maxindex[1]]

        return hit


    @staticmethod
    def attribution_localization(attribution, binary_mask, weighted=False):
        """Implements the Attribution Localization Quantification Method as described in
        Kohlbrenner et. al., 2020, Towards Best Practice in Explaining Neural Network Decisions with LRP
        """

        assert not np.all((attribution < 0.0))
        assert attribution.shape == binary_mask.shape
        assert np.any(binary_mask)

        # filter positive attribution values
        attribution[attribution < 0.0] = 0.0

        # compute inside/outside ratio
        inside_attribution = np.sum(attribution[binary_mask])
        total_attribution = np.sum(attribution)

        size_bbox = float(np.sum(binary_mask))
        size_data = float(np.shape(binary_mask)[0] * np.shape(binary_mask)[1])
        ratio = size_bbox / size_data

        if inside_attribution / total_attribution > 1.0:
            print(
                "inside explanation {} greater than total explanation {}".format(
                    inside_attribution, total_attribution
                )
            )

        inside_attribution_ratio = inside_attribution / total_attribution

        if not weighted:
            return inside_attribution_ratio

        else:
            weighted_inside_attribution_ratio = inside_attribution_ratio * (
                size_data / size_bbox
            )

            return weighted_inside_attribution_ratio
