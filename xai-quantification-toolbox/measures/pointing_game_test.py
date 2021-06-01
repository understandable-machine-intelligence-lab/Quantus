import numpy as np
from typing import Union
from scipy.ndimage import gaussian_filter

from .base import Measure


class PointingGameTest(Measure):
    """ Implements the Pointing Game as described in
    Zhang et. al., 2018, Top-Down Neural Attention by Exitation Backprop
    """

    def __init__(self, **params):
        self.params = params

        super(PointingGameTest, self).__init__()

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int, None],
                 attributions: Union[np.array, None],
                 gaussian_blur=True,
                 **kwargs):

        ratios, hits = [], []
        # results = []

        if gaussian_blur:
            attributions = [gaussian_filter(attribution, sigma=0.02 * np.max(attribution))
                            for attribution in attributions]    # maybe: np.max(np.abs(attribution))    ?

        # iterate samples
        for s, sample in enumerate(inputs):

            # find index of max value
            maxindex = np.where(attributions[s] == np.max(attributions[s]))

            binary_mask = get_binary_mask(sample, targets[s])   # sample.binary_mask[label]

            assert (binary_mask.shape == attributions[s].shape)

            ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

            # check if maximum of explanation is on target object class

            # case max is at more than one pixel
            if len(maxindex[0]) > 1:
                hit = 0
                for pixel in maxindex:
                    hit = hit or binary_mask[pixel[0], pixel[1]]
            else:
                hit = binary_mask[maxindex[0], maxindex[1]]

            # save ratio and hit ?
            ratios.append(ratio)
            hits.append(hit)
            # results.append((ratio, hit))

        return ratios, hits         # maybe as a dict??
        # return results
