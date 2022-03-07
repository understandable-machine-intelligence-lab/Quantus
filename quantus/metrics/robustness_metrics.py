"""This module contains the collection of robustness metrics to evaluate attribution-based explanations of neural network models."""
from typing import Union, List, Dict

import numpy as np
from tqdm import tqdm

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_std (float): The amount of noise added, default=0.1.
            perturb_mean (float): The mean of noise added, default=0.0.
            nr_samples (integer): The number of samples iterated, default=200.
            norm_numerator (callable): Function for norm calculations on the numerator, default=distance_euclidean.
            norm_denominator (callable): Function for norm calculations on the denominator, default=distance_euclidean.
            perturb_func (callable): Input perturbation function, default=gaussian_noise.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=lipschitz_constant.
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
        self.perturb_std = self.kwargs.get("perturb_std", 0.1)
        self.perturb_mean = self.kwargs.get("perturb_mean", 0.0)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", distance_euclidean)
        self.norm_denominator = self.kwargs.get("norm_denominator", distance_euclidean)
        self.perturb_func = self.kwargs.get("perturb_func", gaussian_noise)
        self.similarity_func = self.kwargs.get("similarity_func", lipschitz_constant)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_noise_zero(noise=self.perturb_std)
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        # Get explanation function and make asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for ix, (x, y, a) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            similarity_max = 0.0
            for i in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_perturbed,
                    targets=y,
                    **self.kwargs,
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)
                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                # Measure similarity.
                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed.flatten(),
                )

                if similarity > similarity_max:
                    similarity_max = similarity

            # Append similarity score.
            self.last_results.append(similarity_max)

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
            perturb_radius (float): Perturbation radius, default=0.2.
            nr_samples (integer): The number of samples iterated, default=200.
            norm_numerator (callable): Function for norm calculations on the numerator, default=fro_norm.
            norm_denominator (callable): Function for norm calculations on the denominator, default=fro_norm.
            perturb_func (callable): Input perturbation function, default=uniform_sampling.
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
        self.perturb_radius = self.kwargs.get("perturb_radius", 0.2)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)

        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'perturb_radius', the number of samples "
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
            warn_noise_zero(noise=self.perturb_radius)
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        # Get explanation function and make asserts.
        assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for ix, (x, y, a) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            sensitivities_norm_max = 0.0
            for _ in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_perturbed,
                    targets=y,
                    **self.kwargs,
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=False.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            perturb_radius (float): Perturbation radius, default=0.2.
            nr_samples (integer): The number of samples iterated, default=200.
            norm_numerator (callable): Function for norm calculations on the numerator, default=fro_norm.
            norm_denominator (callable): Function for norm calculations on the denominator, default=fro_norm.
            perturb_func (callable): Input perturbation function, default=uniform_sampling.
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
        self.perturb_radius = self.kwargs.get("perturb_radius", 0.2)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "amount of noise added 'perturb_radius', the number of samples "
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
            warn_noise_zero(noise=self.perturb_radius)
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for ix, (x, y, a) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            self.sub_results = []
            for _ in range(self.nr_samples):

                # Perturb input.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Generate explanation based on perturbed input x.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_perturbed,
                    targets=y,
                    **self.kwargs,
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                sensitivities = self.similarity_func(
                    a=a.flatten(), b=a_perturbed.flatten()
                )
                sensitivities_norm = self.norm_numerator(
                    a=sensitivities
                ) / self.norm_denominator(a=x.flatten())

                self.sub_results.append(sensitivities_norm)

            # Append average sensitivity score.
            self.last_results.append(float(np.mean(self.sub_results)))

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
            abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): Callable that plots the metrics result.
            disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
            img_size (integer): Square image dimensions, default=224.
            patch_size (integer): The patch size for masking, default=7.
            perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            nr_steps (integer): The number of steps to iterate over, default=28.
            perturb_func (callable): Input perturbation function, default=translation_x_direction.
            similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=lipschitz_constant.
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
        self.img_size = self.kwargs.get("img_size", 224)
        self.patch_size = self.kwargs.get("patch_size", 7)
        self.nr_patches = int((self.img_size / self.patch_size) ** 2)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.nr_steps = self.kwargs.get("nr_steps", 28)
        self.dx = self.img_size // self.nr_steps
        self.perturb_func = self.kwargs.get("perturb_func", translation_x_direction)
        self.similarity_func = self.kwargs.get("similarity_func", lipschitz_constant)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_patch_size(patch_size=self.patch_size, img_size=self.img_size)

    def __call__(
        self,
        model: ModelInterface,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
                nr_channels (integer): Number of images, default=second dimension of the input.
                img_size (integer): Image dimension (assumed to be squared), default=last dimension of the input.
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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """
        # Reshape input batch to channel first order:
        self.channel_first = kwargs.get("channel_first", get_channel_first(x_batch))
        x_batch_s = get_channel_first_batch(x_batch, self.channel_first)

        # Wrap the model into an interface.
        if model:
            model = get_wrapped_model(model, self.channel_first)

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch_s)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch_s)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = {k: None for k in range(len(x_batch_s))}

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

        # use tqdm progressbar if not disabled
        if not self.display_progressbar:
            iterator = enumerate(zip(x_batch_s, y_batch, a_batch))
        else:
            iterator = tqdm(
                enumerate(zip(x_batch_s, y_batch, a_batch)), total=len(x_batch_s)
            )

        for ix, (x, y, a) in iterator:

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            sub_results = {k: [] for k in range(self.nr_patches + 1)}

            for step in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(
                    x,
                    **{
                        **{
                            "perturb_dx": (step + 1) * self.dx,
                            "perturb_baseline": self.perturb_baseline,
                        },
                        **self.kwargs,
                    },
                )

                # Generate explanations on perturbed input.
                a_perturbed = explain_func(
                    model=model.get_model(),
                    inputs=x_perturbed,
                    targets=y,
                    **self.kwargs,
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalise:
                    a_perturbed = self.normalise_func(a_perturbed)

                # Store the prediction score as the last element of the sub_self.last_results dictionary.
                x_input = model.shape_input(
                    x_perturbed, self.img_size, self.nr_channels
                )
                y_pred = float(
                    model.predict(x_input, softmax_act=False, **self.kwargs)[:, y]
                )

                sub_results[self.nr_patches].append(y_pred)

                ix_patch = 0
                for i_x, top_left_x in enumerate(
                    range(0, self.img_size, self.patch_size)
                ):
                    for i_y, top_left_y in enumerate(
                        range(0, self.img_size, self.patch_size)
                    ):
                        a_perturbed_patch = a_perturbed[
                            :,
                            top_left_x : top_left_x + self.patch_size,
                            top_left_y : top_left_y + self.patch_size,
                        ]
                        if self.abs:
                            a_perturbed_patch = np.abs(a_perturbed_patch.flatten())

                        if self.normalise:
                            a_perturbed_patch = self.normalise_func(
                                a_perturbed_patch.flatten()
                            )

                        # Sum attributions for patch.
                        patch_sum = float(sum(a_perturbed_patch))
                        sub_results[ix_patch].append(patch_sum)
                        ix_patch += 1

            self.last_results[ix] = sub_results

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
