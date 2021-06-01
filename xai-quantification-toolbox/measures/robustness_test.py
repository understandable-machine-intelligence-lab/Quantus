import numpy as np
from typing import Union

from .base import Measure


class RobustnessTest(Measure):
    """ Implements basis for all Robustness Test Measures.
    Continuity as in Montavon et al., 2018
    """

    def __init__(self, **params):
        self.params = params
        self.perturbation_function = params.get("perturbation_function", None)

        super(RobustnessTest, self).__init__()

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int, None],
                 attributions: Union[np.array, None],
                 **kwargs):

        results = []

        for s, sample in enumerate(inputs):

            attribution = attributions[s]

            perturbed_sample = self.perturbation_function(sample)
            perturbed_attribution = model.attribute(perturbed_sample, ...)

            diff = np.linalg.norm(attribution - perturbed_attribution, ord=1) / \
                np.linalg.norm(sample - perturbed_sample)

            results.append(diff)

        return results
