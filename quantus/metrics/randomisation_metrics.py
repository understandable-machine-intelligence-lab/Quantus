"""This module contains the collection of randomisation metrics to evaluate attribution-based explanations of neural network models."""
import warnings
from typing import Callable, Dict, List, Union

import numpy as np
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.similar_func import correlation_spearman, ssim


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
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            similarity_func (callable): similarity function applied to compare input and perturbed input,
            default=correlation_spearman
            layer_order (string): indicates whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent"
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.seed = self.kwargs.get("seed", 42)
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        asserts.assert_layer_order(layer_order=self.layer_order)
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "similarity metric 'similarity_func' and the order of "
                    "the layer randomisation 'layer_order'"
                ),
                citation=(
                    "Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. "
                    "'Sanity Checks for Saliency Maps.' arXiv preprint,"
                    " arXiv:1810.073292v3 (2018)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
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
            >> metric = ModelParameterRandomisation(abs=True, normalise=False)
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

        self.last_results = {}

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        asserts.assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Create progress bar if desired.
        # Due to the nested for-loops and the requirement of a single progressbar,
        # manual updating will be performed at the end of each inner iteration.
        if self.display_progressbar:
            n_layers = len(
                list(model.get_random_layer_generator(order=self.layer_order))
            )
            n_iterations = n_layers * len(a_batch)
            pbar = tqdm(
                total=n_iterations,
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for layer_name, random_layer_model in model.get_random_layer_generator(
            order=self.layer_order, seed=self.seed
        ):

            similarity_scores = []

            # Generate an explanation with perturbed model.
            a_perturbed = explain_func(
                model=random_layer_model, inputs=x_batch, targets=y_batch, **self.kwargs
            )

            for ix, (a, a_per) in enumerate(zip(a_batch, a_perturbed)):

                if self.normalise:
                    a = self.normalise_func(a)
                    a_per = self.normalise_func(a_per)

                if self.abs:
                    a = np.abs(a)
                    a_per = np.abs(a_per)

                # Compute distance measure.
                similarity = self.similarity_func(a_per.flatten(), a.flatten())

                similarity_scores.append(similarity)

                # Update progress bar if desired.
                if self.display_progressbar:
                    pbar.update(1)

            # Save similarity scores in a dictionary.
            self.last_results[layer_name] = similarity_scores

        # Close progress bar if desired.
        if self.display_progressbar:
            pbar.close()

        if self.return_aggregate:
            print(
                "A 'return_aggregate' functionality is not implemented for this metric."
            )

        self.all_results.append(self.last_results)

        return self.last_results


class RandomLogit(Metric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.

    References:
        1) Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. "When Explanations Lie: Why Many Modified BP
        Attributions Fail."arXiv preprint, arXiv:1912.09818v6 (2020)
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            args: a arguments (optional)
            kwargs: a dictionary of key, value pairs (optional)
            abs: a bool stating if absolute operation should be taken on the attributions
            normalise: a bool stating if the attributions should be normalised
            normalise_func: a Callable that make a normalising transformation of the attributions
            default_plot_func: a Callable that plots the metrics result
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate: a bool if an aggregated score should be produced for the metric over all instances
            aggregate_func: a Callable to aggregate the scores per instance to one float
            last_results: a list containing the resulting scores of the last metric instance call
            all_results: a list containing the resulting scores of all the calls made on the metric instance
            similarity_func (callable): similarity function applied to compare input and perturbed input,
            default=ssim
            num_classes (integer): number of prediction classes in the input, default=1000
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.num_classes = self.kwargs.get("num_classes", 1000)
        self.seed = self.kwargs.get("seed", 42)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("similarity metric 'similarity_func'"),
                citation=(
                    "Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. 'When Explanations Lie: "
                    "Why Many Modified BP Attributions Fail.' arXiv preprint, "
                    "arXiv:1912.09818v6 (2020)"
                ),
            )

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        s_batch: Union[np.array, None] = None,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of some explanations (a_batch) with respect to some input data
        (x_batch), some output labels (y_batch) and a torch model (model).

        Parameters
            model: a torch model e.g., torchvision.models that is subject to explanation
            x_batch: a np.ndarray which contains the input data that are explained
            y_batch: a np.ndarray which contains the output labels that are explained
            a_batch: a Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations
            args: Arguments (optional)
            kwargs: Keyword arguments (optional)
                channel_first (boolean): Indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): Callable generating attributions, default=Callable.

        Returns
            last_results: a list of float(s) with the evaluation outcome of concerned batch

        Examples
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
            >> metric = RandomLogit(abs=True, normalise=False)
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

        self.last_results = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        asserts.assert_explain_func(explain_func=explain_func)

        if a_batch is None:
            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            # Randomly select off-class labels.
            np.random.seed(self.seed)
            y_off = np.array(
                [
                    np.random.choice(
                        [y_ for y_ in list(np.arange(0, self.num_classes)) if y_ != y]
                    )
                ]
            )

            # Explain against a random class.
            a_perturbed = explain_func(
                model=model.get_model(),
                inputs=np.expand_dims(x, axis=0),
                targets=y_off,
                **self.kwargs,
            )

            if self.normalise:
                a_perturbed = self.normalise_func(a_perturbed)

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            self.last_results.append(
                self.similarity_func(a.flatten(), a_perturbed.flatten())
            )

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results
