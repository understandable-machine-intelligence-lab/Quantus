"""This module contains the collection of randomization metrics to evaluate attribution-based explanations of neural network models."""
import numpy as np
import random
from typing import Union, List, Dict
from .base import Metric
from ..helpers.utils import *
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *
from ..helpers.normalize_func import *


class ModelParameterRandomization(Metric):
    """
    TODO. Rewrite docstring.

    Implements the Model Parameter Randomization Method as described in
    Adebayo et. al., 2018, Sanity Checks for Saliency Maps
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.normalize = kwargs.get("normalize", True)
        # explain_func = self.kwargs.get("explain_func", Callable)
        self.last_results = {}
        self.all_results = []

        # Asserts and checks.
        assert_layer_order(layer_order=self.layer_order)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_explain_func(explain_func=explain_func)

        # Save state_dict.
        original_parameters = model.state_dict()

        for layer_name, layer in get_layers(model, order=self.layer_order):

            similarity_scores = []

            if self.layer_order == "independent":
                model.load_state_dict(original_parameters)

            # Randomize layer.
            layer.reset_parameters()

            # Generate an explanation with perturbed model.
            a_perturbed = explain_func(
                model=model, inputs=x_batch, targets=y_batch, **self.kwargs
            )

            for sample, (a, a_per) in enumerate(zip(a_batch, a_perturbed)):

                if self.abs:
                    a = np.abs(a)
                    a_per = np.abs(a_per)

                if self.normalize:
                    a = self.normalize_func(a)
                    a_per = self.normalize_func(a_per)

                # Compute distance measure.
                distance = self.similarity_func(a_per.flatten(), a.flatten())

                similarity_scores.append(distance)

            # Save similarity scores in a dictionary.
            self.last_results[layer_name] = similarity_scores

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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.default_plot_func = Callable
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.num_classes = self.kwargs.get("num_classes", 1000)
        self.max_class = self.kwargs.get("max_class", 10)
        # explain_func = self.kwargs.get("explain_func", Callable)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        # TODO. Add here.

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = [dict() for _ in x_batch]

        if a_batch is None:
            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)
        assert_explain_func(explain_func=explain_func)

        if self.abs:
            a_batch = np.abs(a_batch)

        if self.normalize:
            a_batch = self.normalize_func(a_batch)

        # Randomly select off class labels.
        if isinstance(y_batch, np.ndarray):

            y_batch_off = []

            for idx in y_batch:
                y_range = list(np.arange(0, self.num_classes))
                y_range.remove(idx)
                y_batch_off.append(random.choice(y_range))

            y_batch_off = np.array(y_batch_off)

        else:

            y_range = list(np.arange(0, self.num_classes))
            y_range.remove(y_batch)
            y_batch_off = np.array(
                [random.choice(y_range) for x in range(x_batch.shape[0])]
            )

        a_perturbed = explain_func(
            model=model,
            inputs=x_batch,
            targets=y_batch_off,
            **self.kwargs,
        )

        if self.abs:
            a_perturbed = np.abs(a_perturbed)

        if self.normalize:
            a_perturbed = self.normalize_func(a_perturbed)

        self.last_results = [
            self.similarity_func(a.flatten(), a_per.flatten())
            for a, a_per in zip(a_batch, a_perturbed)
        ]

        self.all_results.append(self.last_results)

        return self.last_results
