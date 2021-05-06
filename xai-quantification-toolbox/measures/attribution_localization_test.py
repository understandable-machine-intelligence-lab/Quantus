import numpy as np
from typing import Union

from .base import Measure


class AttributionLocalizationTest(Measure):
    """ Implements the Attribution Localization Quantification Method as described in
    Kohlbrenner et. al., 2020, Towards Best Practice in Explaining Neural Network Decisions with LRP
    """

    def __init__(self, **params):
        self.params = params

        super(AttributionLocalizationTest, self).__init__()

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int, None],
                 attributions: Union[np.array, None],
                 **kwargs):

        results = []

        for s, sample in enumerate(inputs):

            binary_mask = get_binary_mask(sample, targets[s])  # sample.binary_mask[label]

            # check on any positive value in explanation
            if not np.all((attributions[s] < 0.0)):

                # preprocess attribution (different methods possible), maybe as optional parameter ??
                attribution = np.max(attributions[s], axis=2)

                # filter positive attribution values
                attribution[attribution < 0.0] = 0.0

                # compute inside - total relevance ratios
                binary_mask = binary_mask.astype(bool)[:, :, 0]         # in what form is the binary mask ??
                # binary_mask = np.repeat(binary_mask, 3, 2)

                assert attribution.shape == binary_mask.shape

                if not np.any(binary_mask):
                    print("no True values in binary mask discovered: {}, class: {}".format(s, targets[s]))

                    inside_attribution = np.sum(attribution[binary_mask])
                    total_attribution = np.sum(attribution)

                    size_bbox = float(np.sum(binary_mask))
                    size_data = float(np.shape(binary_mask)[0] * np.shape(binary_mask)[1])
                    ratio = size_bbox / size_data

                    if inside_attribution / total_attribution > 1.0:
                        print("inside explanation {} greater than total explanation {}".format(inside_attribution,
                                                                                               total_attribution))
                    # raise ValueError("inside explanation {} greater than total explanation {}".format(inside_explanation, total_explanation))

                    inside_attribution_ratio = inside_attribution / total_attribution
                    weighted_inside_attribution_ratio = (inside_attribution_ratio) * (size_data / size_bbox)

                    results.append((ratio, inside_attribution_ratio, weighted_inside_attribution_ratio))

            else:
                print("No positive attributed values for sample {} and target {}".format(s, targets[s]))
