import numpy as np
from typing import Union
from .base import Measure
from ..helpers.similarity_functions import *

SIMILARITY_FUNCTIONS = {"correlation_spearman": correlation_spearman,
                        "correlation_pearson": correlation_pearson,
                        "correlation_kendall_tau": correlation_kendall_tau,
                        "distance_euclidean": distance_euclidean,
                        "distance_manhattan": distance_manhattan,
                        "distance_chebyshev": distance_chebyshev,
                        "lipschitz_constant": lipschitz_constant,
                        "cosine": cosine,
                        "ssim": ssim,
                        "mse": mse}


class RobustnessTest(Measure):
    """ Implements basis for all Robustness Test Measures.
    Continuity as in Montavon et al., 2018
    """

    def __init__(self, **params):
        self.params = params
        self.perturbation_function = params.get("perturbation_function", None)
        self.similarity_function = params.get("similarity_function", None)

        assert self.similarity_function in SIMILARITY_FUNCTIONS, "Specify a 'similarity_function' that exist in file."

        super(RobustnessTest, self).__init__()

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int],
                 attributions: Union[np.array, None],
                 **kwargs):
        assert "xai_method" in kwargs, "To run RobustnessTest specify 'xai_method' (str) e.g., 'gradient'."
        assert np.shape(inputs) == np.shape(attributions), "Inputs and attributions should have same shape."

        if attributions is None:
            attributions = model.attribute(inputs, targets, kwargs.get("xai_method", "gradient"))

        # Reshape images and attributions as vectors.
        inputs = np.reshape(inputs, (-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224), kwargs.get("nr_channels", 3)))
        attributions = np.reshape(attributions, (-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224), 1))

        results = []

        for index, (input, target) in enumerate(zip(inputs, targets)):

            # Make explanation based on perturbed input.
            input_perturbed = self.perturbation_function(input)
            attribution = attributions[index]
            attribution_perturbed = model.attribute(batch=input_perturbed, neuron_selection=target, xai_method=kwargs.get("xai_method", "gradient"))
            results.append(self.similarity_function(a=attribution, b=attribution_perturbed, c=input, d=input_perturbed))

        return results