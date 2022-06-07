"""This module contains the collection of focalization metrics to evaluate attribution-based explanations of neural network models."""
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import plotting
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative


class Focus(Metric):
    """
    Implementation of Focus evaluation strategy by Arias et. al. 2022
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plotting.plot_region_perturbation_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)

        self.mosaic_shape = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "no parameter! No parameters means nothing to be sensistive on."
                ),
                citation=(
                    "Arias-Duart, Anna, et al. 'Focus! Rating XAI Methods and Finding Biases.'"
                    "arXiv:2109.15035 (2021)"
                ),
            )

    def __call__(
            self,
            model: ModelInterface,
            x_batch: np.array,
            y_batch: np.array,
            a_batch: Union[np.array, None],
            p_batch: List[tuple],
            *args,
            **kwargs,
    ) -> Dict[int, List[float]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input mosaics that are explained
            y_batch: a np.ndarray which contains the target classes that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            p_batch: a List[tuple] which contains the positions of the target class within the mosaic. Each tuple
                     contains 0/1 values referring to (top_left, top_right, bottom_left, bottom_right).
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dictionary whose values contains a list of float(s) with the evaluation outcome of concerned batch

        Examples
            # TODO: Think about an example using an small dataset mosaics?
            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/mnist"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Focus(abs=False, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface.
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **self.kwargs
            )
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        img_size = a_batch[0, :, :].size

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, p_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, p_batch)), total=len(x_batch_s)
            )

        self.mosaic_shape = a_batch[0].shape
        for sample, (x, y, a, p) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # a is the explainability mosaic with shape (width, height)
            # TODO: Check this assumption

            total_positive_relevance = np.sum(a[a > 0])
            target_positive_relevance = 0
            quadrant_functions_list = [self.quadrant_top_left, self.quadrant_top_right, self.quadrant_bottom_left, self.quadrant_bottom_right]
            for quadrant_p, quadrant_func in zip(p, quadrant_functions_list):
                if not bool(quadrant_p):
                    continue
                quadrant_relevance = quadrant_func(a)
                target_positive_relevance += np.sum(quadrant_relevance[quadrant_relevance > 0])

            focus_score = target_positive_relevance / total_positive_relevance
            self.last_results.append(focus_score)

        self.all_results.append(self.last_results)
        return self.all_results

    def quadrant_top_left(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[0:int(self.mosaic_shape[0] / 2), 0:int(self.mosaic_shape[1] / 2)]
        return quandrant_hmap

    def quadrant_top_right(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[0:int(self.mosaic_shape[0] / 2), int(self.mosaic_shape[1] / 2):self.mosaic_shape[1]]
        return quandrant_hmap

    def quadrant_bottom_left(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[int(self.mosaic_shape[0] / 2):self.mosaic_shape[0], 0:int(self.mosaic_shape[1] / 2)]
        return quandrant_hmap

    def quadrant_bottom_right(self, hmap: np.ndarray) -> np.ndarray:
        quandrant_hmap = hmap[int(self.mosaic_shape[0] / 2):self.mosaic_shape[0], int(self.mosaic_shape[1] / 2):self.mosaic_shape[1]]
        return quandrant_hmap