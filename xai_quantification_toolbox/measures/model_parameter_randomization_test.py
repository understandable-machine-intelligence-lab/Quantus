import numpy as np
from typing import Union

from .base import Measure


class ModelParameterRandomizationTest(Measure):
    """Implements the Model Parameter Randomization Method as described in
    Adebayo et. al., 2018, Sanity Checks for Saliency Maps
    """

    def __init__(self, **params):
        self.params = params
        self.distance_measure = params.get("distance_measure", "spearman")
        self.layer_order = params.get("layer_order", "independent")

        super(ModelParameterRandomizationTest, self).__init__()

    def __call__(
        self,
        model,
        inputs: np.array,
        targets: Union[np.array, int, None],
        attributions: Union[np.array, None],
        **kwargs
    ):

        results = []

        for layer in get_layers(model, order=(self.layer_order == "top_down")):

            layer_results = []

            model = randomize_layer(
                model, layer, independent=(self.layer_order == "independent")
            )

            for s, sample in enumerate(inputs):

                original_attribution = attributions[s]

                modified_attribution = model.attribute(...)

                # normalize attributions
                original_attribution /= np.max(np.abs(original_attribution))
                modified_attribution /= np.max(np.abs(modified_attribution))

                # compute distance measure / ToDo: ensure computation of multiple distance measures at once
                distance = self.distance_measure(
                    modified_attribution, original_attribution
                )

                layer_results.append(distance)

            results.append(layer_results)  # how to save results ??

        return results
