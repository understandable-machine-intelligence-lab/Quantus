import numpy as np
from typing import Union

from .base import Measure
from ..helpers import explain


def get_layers(model, order="top_down"):
    """ Checks a pytorch model for randomizable layers and returns them in a dict. """
    layers = [tup[1] for tup in model.named_modules() if hasattr(tup[1], "reset_parameters")]

    if order == "top_down":
        return layers[::-1]
    else:
        return layers


class ModelParameterRandomizationTest(Measure):
    """Implements the Model Parameter Randomization Method as described in
    Adebayo et. al., 2018, Sanity Checks for Saliency Maps
    """

    def __init__(self, **params):
        self.params = params
        self.distance_measure = params.get("distance_measure", "spearman")
        self.layer_order = params.get("layer_order", "independent")
        self.explanation_func = params.get("explanation_func", "Saliency")

        assert self.layer_order in ["top_down", "bottom_up", "independent"]

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

        # save state_dict
        original_parameters = model.state_dict()

        for layer in get_layers(model, order=(self.layer_order == "top_down")):

            layer_results = []

            if self.layer_order == "independent":
                model.load_state_dict(original_parameters)

            # randomize layer
            layer.reset_parameters()

            modified_attributions = explain(model, inputs, targets, explanation_func=self.explanation_func)

            for original_attribution, modified_attribution in zip(attributions, modified_attributions):

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
