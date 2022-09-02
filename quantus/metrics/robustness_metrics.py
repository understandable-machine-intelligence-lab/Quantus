"""This module contains the collection of robustness metrics to evaluate attribution-based explanations of neural network models."""
import itertools
import warnings
from typing import Callable, Dict, List, Union

import numpy as np
from tqdm import tqdm

from .base import Metric
from ..helpers import asserts
from ..helpers import perturb_func
from ..helpers import similar_func
from ..helpers import utils
from ..helpers import warn_func
from ..helpers.asserts import attributes_check
from ..helpers.model_interface import ModelInterface
from ..helpers.norm_func import fro_norm
from ..helpers.normalise_func import normalise_by_negative
from ..helpers.discretise_func import top_n_sign


class LocalLipschitzEstimate(Metric):
    """
    Implementation of the Local Lipschitz Estimated (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).

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
            perturb_std (float): The amount of noise added, default=0.1
            perturb_mean (float): The mean of noise added, default=0.0
            nr_samples (integer): The number of samples iterated, default=200
            norm_numerator (callable): Function for norm calculations on the numerator, default=distance_euclidean
            norm_denominator (callable): Function for norm calculations on the denominator, default=distance_euclidean
            perturb_func (callable): Input perturbation function, default=gaussian_noise
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=lipschitz_constant
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
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get(
            "norm_numerator", similar_func.distance_euclidean
        )
        self.norm_denominator = self.kwargs.get(
            "norm_denominator", similar_func.distance_euclidean
        )
        self.perturb_func = self.kwargs.get("perturb_func", perturb_func.gaussian_noise)
        self.perturb_std = self.kwargs.get("perturb_std", 0.1)
        self.perturb_mean = self.kwargs.get("perturb_mean", 0.0)
        self.similarity_func = self.kwargs.get(
            "similarity_func", similar_func.lipschitz_constant
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'perturb_std', the number of samples iterated "
                    "over 'nr_samples', the function to perturb the input 'perturb_func',"
                    " the similarity metric 'similarity_func' as well as norm "
                    "calculations on the numerator and denominator of the lipschitz "
                    "equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'On the robustness of "
                    "interpretability methods.' arXiv preprint arXiv:1806.08049 (2018). and "
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint "
                    "arXiv:1806.07538 (2018)"
                ),
            )
            warn_func.warn_noise_zero(noise=self.perturb_std)

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
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = LocalLipschitzEstimate(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
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

        # Get explanation function and make asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

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

            similarity_max = 0.0
            for i in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=np.arange(0, x.size),
                    indexed_axes=np.arange(0, x.ndim),
                    **self.kwargs,
                )
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_input,
                    targets=y,
                    **self.kwargs,
                )

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                # Measure similarity.
                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed.flatten(),
                    **self.kwargs,
                )

                if similarity > similarity_max:
                    similarity_max = similarity

            # Append similarity score.
            self.last_results.append(similarity_max)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class MaxSensitivity(Metric):
    """
    Implementation of max-sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the maximum sensitivity is captured.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            lower_bound (float): Lower Bound of Perturbation, default=0.2.
            upper_bound (None, float): Upper Bound of Perturbation, default=None.
            nr_samples (integer): The number of samples iterated, default=200.
            norm_numerator (callable): Function for norm calculations on the numerator, default=fro_norm.
            norm_denominator (callable): Function for norm calculations on the denominator, default=fro_norm.
            perturb_func (callable): Input perturbation function, default=uniform_noise.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=difference.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", perturb_func.uniform_noise)
        self.lower_bound = self.kwargs.get("lower_bound", 0.2)
        self.upper_bound = self.kwargs.get("upper_bound", None)
        self.similarity_func = self.kwargs.get(
            "similarity_func", similar_func.difference
        )

        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn_func.warn_noise_zero(noise=self.lower_bound)

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
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = MaxSensitivity(abs=True, normalise=False)
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

        # Get explanation function and make asserts.
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

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

            sensitivities_norm_max = 0.0
            for _ in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=np.arange(0, x.size),
                    indexed_axes=np.arange(0, x.ndim),
                    **self.kwargs,
                )
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_input,
                    targets=y,
                    **self.kwargs,
                )

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                # Measure sensitivity.
                sensitivities = self.similarity_func(
                    a=a.flatten(), b=a_perturbed.flatten()
                )
                sensitivities_norm = self.norm_numerator(
                    a=sensitivities
                ) / self.norm_denominator(a=x.flatten())

                if sensitivities_norm > sensitivities_norm_max:
                    sensitivities_norm_max = sensitivities_norm

            # Append max sensitivity score.
            self.last_results.append(sensitivities_norm_max)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class AvgSensitivity(Metric):
    """
    Implementation of avg-sensitivity by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuring how explanations
    change under slight perturbation - the average sensitivity is captured.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

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
            lower_bound (float): lower Bound of Perturbation, default=0.2
            upper_bound (None, float): upper Bound of Perturbation, default=None
            nr_samples (integer): the number of samples iterated, default=200.
            norm_numerator (callable): function for norm calculations on the numerator, default=fro_norm.
            norm_denominator (callable): function for norm calculations on the denominator, default=fro_norm.
            perturb_func (callable): input perturbation function, default=uniform_noise.
            similarity_func (callable): similarity function applied to compare input and perturbed input,
            default=difference.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", perturb_func.uniform_noise)
        self.lower_bound = self.kwargs.get("lower_bound", 0.2)
        self.upper_bound = self.kwargs.get("upper_bound", None)
        self.similarity_func = self.kwargs.get(
            "similarity_func", similar_func.difference
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'lower_bound' and 'upper_bound', the number of samples "
                    "iterated over 'nr_samples', the function to perturb the input "
                    "'perturb_func', the similarity metric 'similarity_func' as well as "
                    "norm calculations on the numerator and denominator of the sensitivity"
                    " equation i.e., 'norm_numerator' and 'norm_denominator'"
                ),
                citation=(
                    "Yeh, Chih-Kuan, et al. 'On the (in) fidelity and sensitivity for explanations"
                    ".' arXiv preprint arXiv:1901.09392 (2019)"
                ),
            )
            warn_func.warn_noise_zero(noise=self.lower_bound)

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
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

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
            >> metric = AvgSensitivity(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
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
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

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

            self.sub_results = []
            for _ in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=np.arange(0, x.size),
                    indexed_axes=np.arange(0, x.ndim),
                    **self.kwargs,
                )
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_input,
                    targets=y,
                    **self.kwargs,
                )

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                sensitivities = self.similarity_func(
                    a=a.flatten(), b=a_perturbed.flatten()
                )
                sensitivities_norm = self.norm_numerator(
                    a=sensitivities
                ) / self.norm_denominator(a=x.flatten())

                self.sub_results.append(sensitivities_norm)

            # Append average sensitivity score.
            self.last_results.append(float(np.mean(self.sub_results)))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class Continuity(Metric):
    """
    Implementation of the Continuity test by Montavon et al., 2018.

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Assumptions:
        - In this implementation, we assume that height and width dimensions are equally sized.
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
            patch_size (integer): the patch size for masking, default=7.
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black"
            nr_steps (integer): the number of steps to iterate over, default=28
            perturb_func (callable): input perturbation function, default=translation_x_direction
            similarity_func (callable): similarity function applied to compare input and perturbed input,
            default=lipschitz_constant
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction
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
        self.last_results = []
        self.all_results = []

        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.patch_size = self.kwargs.get("patch_size", 7)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.nr_steps = self.kwargs.get("nr_steps", 28)
        self.perturb_func = self.kwargs.get(
            "perturb_func", perturb_func.translation_x_direction
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.similarity_func = self.kwargs.get(
            "similarity_func", similar_func.lipschitz_constant
        )
        self.softmax = self.kwargs.get("softmax", False)

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "how many patches to split the input image to 'nr_patches', "
                    "the number of steps to iterate over 'nr_steps', the value to replace"
                    " the masking with 'perturb_baseline' and in what direction to "
                    "translate the image 'perturb_func'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73, 1-15 (2018"
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
    ) -> Dict[int, List[float]]:
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
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            last_results: a dict of pairs of int(s) and list of float(s) with the evaluation outcome of batch

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
            >> metric = Continuity(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """

        # Reshape input batch to channel first order.
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)

        # Wrap the model into an interface
        if model:
            model = utils.get_wrapped_model(model, channel_first)

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }

        # Run deprecation warnings.
        warn_func.deprecation_warnings(self.kwargs)

        self.last_results = {k: None for k in range(len(x_batch_s))}

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

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_patch_size(patch_size=self.patch_size, shape=x_batch_s.shape[2:])
        asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # Get number of patches for input shape (ignore batch and channel dim).
        self.nr_patches = utils.get_nr_patches(
            patch_size=self.patch_size,
            shape=x_batch_s.shape[2:],
            overlap=True,
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        self.dx = np.prod(x_batch_s.shape[2:]) // self.nr_steps
        for ix, (x, y, a) in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            sub_results = {k: [] for k in range(self.nr_patches + 1)}

            for step in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                dx_step = (step + 1) * self.dx
                x_perturbed = self.perturb_func(
                    arr=x,
                    indices=np.arange(0, x.size),
                    indexed_axes=np.arange(0, x.ndim),
                    perturb_dx=dx_step,
                    **self.kwargs,
                )
                x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)

                # Generate explanations on perturbed input.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_input,
                    targets=y,
                    **self.kwargs,
                )

                # Taking the first element, since a_perturbed will be expanded to a batch dimension
                # not expected by the current index management functions.
                a_perturbed = utils.expand_attribution_channel(a_perturbed, x_input)[0]

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                # Store the prediction score as the last element of the sub_self.last_results dictionary.
                y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

                sub_results[self.nr_patches].append(y_pred)

                # Create patches by splitting input into grid.
                axis_iterators = [
                    range(0, x_input.shape[axis], self.patch_size) for axis in a_axes
                ]

                for ix_patch, top_left_coords in enumerate(
                    itertools.product(*axis_iterators)
                ):

                    # Create slice for patch.
                    patch_slice = utils.create_patch_slice(
                        patch_size=self.patch_size,
                        coords=top_left_coords,
                    )

                    a_perturbed_patch = a_perturbed[
                        utils.expand_indices(a_perturbed, patch_slice, a_axes)
                    ]

                    if self.normalise:
                        a_perturbed_patch = self.normalise_func(
                            a_perturbed_patch.flatten()
                        )

                    if self.abs:
                        a_perturbed_patch = np.abs(a_perturbed_patch.flatten())

                    # Sum attributions for patch.
                    patch_sum = float(sum(a_perturbed_patch))
                    sub_results[ix_patch].append(patch_sum)

            self.last_results[ix] = sub_results

        if self.return_aggregate:
            print(
                "A 'return_aggregate' functionality is not implemented for this metric."
            )

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """
        Implements a continuity correlation score (an addition to the original method) to evaluate the
        relationship between change in explanation and change in function output. It can be seen as an
        quantitative interpretation of visually determining how similar f(x) and R(x1) curves are.
        """
        return np.mean(
            [
                self.similarity_func(
                    self.last_results[sample][self.nr_patches],
                    self.last_results[sample][ix_patch],
                )
                for ix_patch in range(self.nr_patches)
                for sample in self.last_results.keys()
            ]
        )


class Consistency(Metric):
    """

    The (global) consistency metric measures the expected local consistency. Local consistency measures the probability
    of the prediction label for a given datapoint coinciding with the prediction labels of other data points that
    the same explanation is being attributed to. For example, if the explanation of a given image is "contains zebra",
    the local consistency metric measures the probability a different image that the explanation "contains zebra" is
    being attributed to having the same prediction label.

    References:
         1) Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. "Framework for Evaluating Faithfulness of Local
            Explanations." arXiv preprint arXiv:2202.00734 (2022).

    Assumptions:
        - A used-defined discreization function is used to discretize continuous explanation spaces.
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
            discretise_func (callable): the explanation space discretisation function, returns hash value of an array;
            arrays with identical elements receive the same hash value, default=top_n_sign
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
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.last_results = []
        self.all_results = []

        self.discretise_func = self.kwargs.get("discretise_func", top_n_sign)

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "Function for discretisation of the explanation space 'discretise_func' (return hash value of"
                    "an np.array used for comparison)."
                ),
                citation=(
                    "Sanjoy Dasgupta, Nave Frost, and Michal Moshkovitz. 'Framework for Evaluating Faithfulness of "
                    "Explanations.' arXiv preprint arXiv:2202.00734 (2022)"
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
    ) -> Dict[int, List[float]]:
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
                device (string): Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
                default=None.

        Returns
            metric: value of the metric.

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

        a_batch_flat = a_batch.reshape(a_batch.shape[0], -1)

        a_labels = np.array(list(map(self.discretise_func, a_batch_flat)))

        # Predict on input.
        x_input = model.shape_input(
            x_batch, x_batch[0].shape, channel_first=True, batch=True
        )
        y_pred_classes = np.argmax(
            model.predict(x_input, softmax_act=True, **self.kwargs), axis=1
        ).flatten()

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch, a_labels))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch, a_labels)),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for ix, (x, y, a, a_label) in iterator:

            pred_a = y_pred_classes[ix]
            same_a = np.argwhere(a == a_label).flatten()
            diff_a = same_a[same_a != ix]
            pred_same_a = y_pred_classes[diff_a]

            if len(same_a) == 0:
                self.last_results.append(0)
            else:
                self.last_results.append(np.sum(pred_same_a == pred_a) / len(diff_a))

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results
