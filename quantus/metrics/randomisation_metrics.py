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
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=correlation_spearman.
            layer_order (string): Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.layer_order = kwargs.get("layer_order", "independent")
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        assert_layer_order(layer_order=self.layer_order)
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)
        # Wrap the model into an interface
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = {}

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        for layer_name, random_layer_model in model.get_random_layer_generator(
            order=self.layer_order
        ):

            similarity_scores = []

            # Generate an explanation with perturbed model.
            a_perturbed = explain_func(
                model=random_layer_model, inputs=x_batch, targets=y_batch, **self.kwargs
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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=ssim.
            num_classes (integer): Number of prediction classes in the input, default=1000.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.similarity_func = self.kwargs.get("similarity_func", ssim)
        self.num_classes = self.kwargs.get("num_classes", 1000)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("similarity metric 'similarity_func'"),
                citation=(
                    "Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. 'When Explanations Lie: "
                    "Why Many Modified BP Attributions Fail.' arXiv preprint, "
                    "arXiv:1912.09818v6 (2020)"
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
        *args,
        **kwargs
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)
        # Wrap the model into an interface
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
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
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Randomly select off-class labels.
            y_off = np.array(
                [
                    random.choice(
                        [y_ for y_ in list(np.arange(0, self.num_classes)) if y_ != y]
                    )
                ]
            )

            # Explain against a random class.
            a_perturbed = explain_func(
                model=model.get_model(),
                inputs=x,
                targets=y_off,
                **self.kwargs,
            )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            if self.normalise:
                a_perturbed = self.normalise_func(a_perturbed)

            self.last_results.append(
                self.similarity_func(a.flatten(), a_perturbed.flatten())
            )

        self.all_results.append(self.last_results)

        return self.last_results
