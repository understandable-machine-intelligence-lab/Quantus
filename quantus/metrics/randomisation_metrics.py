"""This module contains the collection of randomisation metrics to evaluate attribution-based explanations of neural network models."""
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
from ..helpers.normalise_func import *
from ..helpers.warn_func import *


class ModelParameterRandomisation(Metric):
    """
    Implementation of the Model Parameter Randomization Method by Adebayo et. al., 2018.

    The Model Parameter Randomization measures the distance between the original attribution and a newly computed
    attribution throughout the process of cascadingly/independently randomizing the model parameters of one layer
    at a time.

    References:
        1) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. "Sanity Checks for Saliency Maps."
        arXiv preprint, arXiv:1810.073292v3 (2018)

    Assumptions:
        In the original paper multiple distance measures are taken: Spearman rank correlation (with and without abs),
        HOG and SSIM. We have set Spearman as the default value.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.normalise = kwargs.get("normalise", True)
        self.text_warning = (
            "\nThe Model parameter randomisation metric is likely to be sensitive to the choice of "
            "similarity metric 'similarity_func' and the order of layer randomisation 'layer_order'. "
            "Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Adebayo, J., Gilmer, J., Muelly, M., "
            "Goodfellow, I., Hardt, M., and Kim, B. 'Sanity Checks for Saliency Maps.' arXiv preprint, "
            "arXiv:1810.073292v3 (2018).\n"
        )
        self.last_results = {}
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)
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

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

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

                if self.normalise:
                    a = self.normalise_func(a)
                    a_per = self.normalise_func(a_per)

                # Compute distance measure.
                similarity = self.similarity_func(a_per.flatten(), a.flatten())

                similarity_scores.append(similarity)

            # Save similarity scores in a dictionary.
            self.last_results[layer_name] = similarity_scores

        self.all_results.append(self.last_results)

        return self.last_results


class RandomLogit(Metric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation regarding
    a randomly chosen non-target class.

    References:
        1) Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. "When Explanations Lie: Why Many Modified BP
        Attributions Fail."arXiv preprint, arXiv:1912.09818v6 (2020)
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.default_plot_func = Callable
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.num_classes = self.kwargs.get("num_classes", 1000)
        self.text_warning = (
            "\nThe Random Logit metric is likely to be sensitive to the choice of "
            "similarity metric 'similarity_func'."
            "Go over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. To view all relevant hyperparameters call .get_params of the "
            "metric instance. For further reading, please see: Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. "
            "'When Explanations Lie: Why Many Modified BP Attributions Fail.' arXiv preprint, "
            "arXiv:1912.09818v6 (2020).\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalise:
            warn_normalise_abs(normalise=self.normalise, abs=self.abs)

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

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:
            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        if self.abs:
            a_batch = np.abs(a_batch)

        if self.normalise:
            a_batch = self.normalise_func(a_batch)

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

        if self.normalise:
            a_perturbed = self.normalise_func(a_perturbed)

        # Check this.
        self.last_results = [
            self.similarity_func(a.flatten(), a_per.flatten())
            for a, a_per in zip(a_batch, a_perturbed)
        ]

        self.all_results.append(self.last_results)

        return self.last_results
