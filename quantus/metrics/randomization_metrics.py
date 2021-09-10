"""This module contains the collection of randomization metrics to evaluate attribution-based explanations of neural network models."""
import numpy as np
import random
from typing import Union
from .base import Metric
from ..helpers.utils import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *


class ModelParameterRandomization(Metric):
    """
    TODO. Rewrite docstring.

    Implements the Model Parameter Randomization Method as described in
    Adebayo et. al., 2018, Sanity Checks for Saliency Maps
    """

    def __init__(self, *args, **kwargs):
    
        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.normalize = kwargs.get("normalize", True)
        self.explain_func = self.kwargs.get("explain_func", None)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        assert_layer_order(layer_order=self.layer_order)
        assert_explain_func(explain_func=self.explain_func)

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
        self.last_results = [dict() for _ in x_batch]

        if a_batch is None:

            # Generate explanations.
            a_batch = self.explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_atts(a_batch=a_batch, x_batch=x_batch)

        # Save state_dict.
        original_parameters = model.state_dict()

        for layer_name, layer in get_layers(model, order=self.layer_order):

            similarity_scores = []

            if self.layer_order == "independent":
                model.load_state_dict(original_parameters)

            # randomize layer
            layer.reset_parameters()

            modified_attributions = self.explain_func(model=model,
                                                      inputs=x_batch,
                                                      targets=y_batch,
                                                      **self.kwargs)

            for att_ori, att_modi in zip(a_batch, modified_attributions):

                # Normalize attributions
                if self.normalize:
                    att_ori /= np.max(np.abs(att_ori))
                    att_modi /= np.max(np.abs(att_modi))

                # Compute distance measure.
                distance = self.similarity_func(att_modi.flatten(), att_ori.flatten())

                similarity_scores.append(distance)

            # Save similarity scores in a dictionary.
            for r, result in enumerate(similarity_scores):
                self.last_results[r][layer_name] = result

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        # TODO. Implement a class method that plots or takes the similarity scores and make something useful from it.
        pass

class RandomLogit(Metric):
    """
    TODO. Rewrite docstring.

    Implements the Random Logit Method as described in
    Sixt et. al., 2020, When Explanations lie
    """

    def __init__(self, *args, **kwargs):
    
        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.max_class = self.kwargs.get("max_class", 10)
        self.explain_func = self.kwargs.get("explain_func", None)
        self.last_results = []
        self.all_results = []

    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
        self.last_results = [dict() for _ in x_batch]

        if a_batch is None:

            # Generate explanations.
            a_batch = self.explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_atts(a_batch=a_batch, x_batch=x_batch)

        # Randomly select off class labels.
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

        a_batch_off = self.explain_func(
            model=model,
            inputs=x_batch,
            targets=y_batch_off,
            **self.kwargs,
        )

        self.last_results = np.array([self.similarity_func(a.flatten(), a_off.flatten())
                            for a, a_off in zip(a_batch, a_batch_off)])

        self.all_results.append(self.last_results)

        return self.last_results
