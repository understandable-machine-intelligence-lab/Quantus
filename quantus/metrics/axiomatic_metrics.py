"""This module contains the collection of axiomatic metrics to evaluate attribution-based explanations of neural network models."""
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
from ..helpers.perturb_func import (
    baseline_replacement_by_indices,
    baseline_replacement_by_shift,
)


class Completeness(Metric):
    """
    Implementation of Completeness test by Sundararajan et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018.

    Attribution completeness asks that the total attribution is proportional to the explainable
    evidence at the output/ or some function of the model output. Or, that the attributions
    add up to the difference between the model output F at the input x and the baseline b.

    References:
        1) Completeness - Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks."
        International Conference on Machine Learning. PMLR, 2017.
        2) Summation to delta - Shrikumar, Avanti, Peyton Greenside, and Anshul Kundaje. "Learning important
        features through propagating activation differences." International Conference on Machine Learning. PMLR, 2017.
        3) Conservation - Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting
        and understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Assumptions:
        This implementation does completeness test against logits, not softmax.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): indicates whether normalise operation is applied on the attribution,
            default=False.
            normalise_func (callable): attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): a Callable that plots the metrics result.
            disable_warnings (boolean): indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate (boolean): indicates if an aggregated score should be produced over all instances.
            aggregate_func (Callable): a Callable to aggregate the scores per instance to one float.
            output_func (callable): a function applied to the difference between the model output at the input and the
            baseline before metric calculation, default=lambda x: x.
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices.
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction.
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
        self.output_func = self.kwargs.get("output_func", lambda x: x)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.softmax = self.kwargs.get("softmax", False)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the function to modify the "
                    "model response 'output_func'"
                ),
                citation=(
                    "Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. 'Axiomatic attribution for "
                    "deep networks.' International Conference on Machine Learning. PMLR, (2017)."
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
    ) -> List[bool]:
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
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = Completeness(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order:
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

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

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
        asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch_s)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            x_baseline = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.kwargs,
            )

            # Predict on input.
            x_input = model.shape_input(x, x.shape, channel_first=True)
            y_pred = float(model.predict(x_input, **self.kwargs)[:, y])

            # Predict on baseline.
            x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
            y_pred_baseline = float(model.predict(x_input, **self.kwargs)[:, y])

            if np.sum(a) == self.output_func(y_pred - y_pred_baseline):
                self.last_results.append(True)
            else:
                self.last_results.append(False)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class NonSensitivity(Metric):
    """
    Implementation of NonSensitivity by Nguyen at el., 2020.

    Non- sensitivity measures if zero-importance is only assigned to features, that the model is not
    functionally dependent on.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Ancona, Marco, et al. "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley
        Values Approximation." arXiv preprint arXiv:1903.10992 (2019).
        3) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            eps (float): Attributions threshold, default=1e-5.
            n_samples (integer): The number of samples to iterate over, default=100.
            abs (boolean): indicates whether absolute operation is applied on the attribution, default=True.
            normalise (boolean): indicates whether normalise operation is applied on the attribution, default=True.
            normalise_func (callable): Attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): a Callable that plots the metrics result.
            disable_warnings (boolean): indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate (boolean): indicates if an aggregated score should be produced over all instances.
            aggregate_func (Callable): a Callable to aggregate the scores per instance to one float.
            perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
            perturb_func (callable): Input perturbation function, default=baseline_replacement_by_indices.
            softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction.
        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.eps = self.kwargs.get("eps", 1e-5)
        self.n_samples = self.kwargs.get("n_samples", 100)
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.display_progressbar = self.kwargs.get("display_progressbar", False)
        self.return_aggregate = self.kwargs.get("return_aggregate", False)
        self.aggregate_func = self.kwargs.get("aggregate_func", np.mean)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.softmax = self.kwargs.get("softmax", True)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the number of samples to iterate"
                    " over 'n_samples' and the threshold value function for the feature"
                    " to be considered having an insignificant contribution to the model"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)."
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
    ) -> List[int]:
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
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = NonSensitivity(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order:
        if "channel_first" in kwargs and isinstance(kwargs["channel_first"], bool):
            channel_first = kwargs.get("channel_first")
        else:
            channel_first = utils.infer_channel_first(x_batch)
        x_batch_s = utils.make_channel_first(x_batch, channel_first)
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

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            asserts.assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model.get_model(),
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
        a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
        a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

        # Asserts.
        asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch_s)
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch_s.shape[2:],
        )

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            a = a.flatten()

            if self.normalise:
                a = self.normalise_func(a)

            if self.abs:
                a = np.abs(a)

            non_features = set(list(np.argwhere(a).flatten() < self.eps))

            vars = []
            for i_ix, a_ix in enumerate(a[:: self.features_in_step]):

                preds = []
                a_ix = a[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ].astype(int)

                for _ in range(self.n_samples):
                    # Perturb input by indices of attributions.
                    x_perturbed = self.perturb_func(
                        arr=x,
                        indices=a_ix,
                        indexed_axes=a_axes,
                        **self.kwargs,
                    )

                    # Predict on perturbed input x.
                    x_input = model.shape_input(
                        x_perturbed, x.shape, channel_first=True
                    )
                    y_pred_perturbed = float(
                        model.predict(x_input, **self.kwargs)[:, y]
                    )
                    preds.append(y_pred_perturbed)

                    vars.append(np.var(preds))

            non_features_vars = set(list(np.argwhere(vars).flatten() < self.eps))
            self.last_results.append(
                len(non_features_vars.symmetric_difference(non_features))
            )

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results


class InputInvariance(Metric):
    """
    Implementation of Completeness test by Kindermans et al., 2017.

    To test for input invaraince, we add a constant shift to the input data and then measure the effect
    on the attributions, the expectation is that if the model show no response, then the explanations should not.

    References:
        Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., Dähne Sven,
        Erhan Dumitru and Kim Been. "THE (UN)RELIABILITY OF SALIENCY METHODS" Article (2017).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args: Arguments (optional)
        kwargs: Keyword arguments (optional)
            abs (boolean): indicates whether absolute operation is applied on the attribution, default=False.
            normalise (boolean): indicates whether normalise operation is applied on the attribution,
            default=False.
            normalise_func (callable): attribution normalisation function applied in case normalise=True,
            default=normalise_by_negative.
            default_plot_func (callable): a Callable that plots the metrics result.
            disable_warnings (boolean): indicates whether the warnings are printed, default=False.
            display_progressbar (boolean): indicates whether a tqdm-progress-bar is printed, default=False.
            return_aggregate (boolean): indicates if an aggregated score should be produced over all instances.
            aggregate_func (Callable): a Callable to aggregate the scores per instance to one float.
            input_shift (integer): shift to the input data, default=-1.
            perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices.
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
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_shift
        )
        self.input_shift = self.kwargs.get("input_shift", -1)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("input shift 'input_shift'"),
                citation=(
                    "Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., "
                    "Dähne Sven, Erhan Dumitru and Kim Been. 'THE (UN)RELIABILITY OF SALIENCY METHODS' Article (2017)."
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
    ) -> List[bool]:
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
                channel_first (boolean): indicates of the image dimensions are channel first, or channel last.
                Inferred from the input shape by default.
                explain_func (callable): a Callable generating attributions, default=Callable.
                device (string): indicates the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu",
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
            >> metric = InputInvariance(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{})
        """
        # Reshape input batch to channel first order:
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
        asserts.assert_attributions(a_batch=a_batch, x_batch=x_batch)

        # Use tqdm progressbar if not disabled.
        if not self.display_progressbar:
            iterator = zip(x_batch_s, y_batch, a_batch)
        else:
            iterator = tqdm(
                zip(x_batch_s, y_batch, a_batch),
                total=len(x_batch_s),
                desc=f"Evaluation of {self.__class__.__name__}",
            )

        for x, y, a in iterator:

            if self.normalise:
                warn_func.warn_normalisation_skipped()

            if self.abs:
                warn_func.warn_absolutes_skipped()

            x_shifted = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                **self.kwargs,
            )
            x_shifted = model.shape_input(x_shifted, x.shape, channel_first=True)
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_shifted)

            # Generate explanation based on shifted input x.
            a_shifted = explain_func(
                model=model.get_model(), inputs=x_shifted, targets=y, **self.kwargs
            )

            # Check if explanation of shifted input is similar to original.
            if (a.flatten() != a_shifted.flatten()).all():
                self.last_results.append(True)
            else:
                self.last_results.append(False)

        if self.return_aggregate:
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results
