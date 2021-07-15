import numpy as np
import random
from typing import Union

from .base import Measure
from ..helpers import explain
from ..helpers.similarity_func import *


def get_layers(model, order="top_down"):
    """ Checks a pytorch model for randomizable layers and returns them in a dict. """
    layers = [tup for tup in model.named_modules() if hasattr(tup[1], "reset_parameters")]

    if order == "top_down":
        return layers[::-1]
    else:
        return layers


class ModelParameterRandomizationTest(Measure):
    """Implements the Model Parameter Randomization Method as described in
    Adebayo et. al., 2018, Sanity Checks for Saliency Maps
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.explanation_func = kwargs.get("explanation_func", "Saliency")

        assert self.layer_order in ["top_down", "bottom_up", "independent"]

        super(ModelParameterRandomizationTest, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):

        results = [dict() for x in x_batch]

        # save state_dict
        original_parameters = model.state_dict()

        for layer_name, layer in get_layers(model, order=(self.layer_order == "top_down")):

            layer_results = []

            if self.layer_order == "independent":
                model.load_state_dict(original_parameters)

            # randomize layer
            layer.reset_parameters()

            modified_attributions = explain(model, x_batch, y_batch, explanation_func=self.explanation_func).cpu().numpy()

            for original_attribution, modified_attribution in zip(a_batch, modified_attributions):

                # normalize attributions
                original_attribution /= np.max(np.abs(original_attribution))
                modified_attribution /= np.max(np.abs(modified_attribution))

                # compute distance measure / ToDo: ensure computation of multiple distance measures at once
                distance = self.similarity_func(
                    modified_attribution.flatten(), original_attribution.flatten()
                )

                layer_results.append(distance)

            for r, result in enumerate(layer_results):
                results[r][layer_name] = result

            # results.append(layer_results)  # how to save results ??

        return results


class RandomLogitTest(Measure):
    """Implements the Random Logit Method as described in
    Sixt et. al., 2020, When Explanations lie
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.max_class = self.kwargs.get("max_class", 10)

        super(RandomLogitTest, self).__init__()

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):

        assert (
                "explanation_func" in kwargs
        ), "To run ContinuityTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            a_batch = explain(
                    model.to(kwargs.get("device", None)),
                    x_batch,
                    y_batch,
                    explanation_func=kwargs.get("explanation_func", "Gradient"),
                    device=kwargs.get("device", None),
                ).cpu().numpy()

        # randomly select off class labels
        if isinstance(y_batch, np.ndarray):
            y_batch_off = []
            for idx in y_batch:
                y_range = list(np.arange(0, self.max_class))
                y_range.remove(idx)
                y_batch_off.append(random.choice(y_range))
            y_batch_off = np.array(y_batch_off)
        else:
            y_range = list(np.arange(0, self.max_class))
            y_range.remove(y_batch)
            y_batch_off = np.array([random.choice(y_range) for x in range(x_batch.shape[0])])

        a_batch_off = explain(
            model.to(kwargs.get("device", None)),
            x_batch,
            y_batch_off,
            explanation_func=kwargs.get("explanation_func", "Gradient"),
            device=kwargs.get("device", None)
        ).cpu().numpy()

        results = np.array([self.similarity_func(a.flatten(), a_off.flatten()) for a, a_off in zip(a_batch, a_batch_off)])

        return results
