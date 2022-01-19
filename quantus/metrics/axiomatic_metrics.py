"""This module contains the collection of axiomatic metrics to evaluate attribution-based explanations of neural network models."""
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

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.output_func = self.kwargs.get("output_func", lambda x: x)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
            args: optional args
            kwargs: optional dict

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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
        assert_attributions(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            x_baseline = self.perturb_func(
                img=x.flatten(),
                **{
                    "indices": np.arange(0, len(x)),
                    "perturb_baseline": self.perturb_baseline,
                },
            )

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            # Predict on baseline.
            with torch.no_grad():
                y_pred_baseline = float(
                    model(
                        torch.Tensor(x_baseline)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            if np.sum(a) == self.output_func(y_pred - y_pred_baseline):
                self.last_results.append(True)
            else:
                self.last_results.append(False)

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
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
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
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
            args: optional args
            kwargs: optional dict

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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
        assert_attributions(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            non_features = set(list(np.argwhere(a).flatten() < self.eps))

            vars = []
            for a_i in range(len(a)):

                preds = []
                for _ in range(self.n_samples):

                    x_perturbed = self.perturb_func(
                        img=x.flatten(),
                        **{
                            "indices": a_i,
                            "perturb_baseline": self.perturb_baseline,
                        },
                    )

                    # Predict on perturbed input x.
                    with torch.no_grad():
                        y_pred_perturbed = float(
                            torch.nn.Softmax()(
                                model(
                                    torch.Tensor(x_perturbed)
                                    .reshape(
                                        1,
                                        self.nr_channels,
                                        self.img_size,
                                        self.img_size,
                                    )
                                    .to(self.kwargs.get("device", None))
                                )
                            )[:, y]
                        )
                        preds.append(y_pred_perturbed)

                    vars.append(np.var(preds))

            non_features_vars = set(list(np.argwhere(vars).flatten() < self.eps))
            self.last_results.append(
                len(non_features_vars.symmetric_difference(non_features))
            )

        self.all_results.append(self.last_results)

        return self.last_results


class InputInvariance(Metric):
    """
    Implementation of Completeness test by Kindermans et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018.

    To test for input invaraince, we add a constant shift to the input data and then measure the effect
    on the attributions, the expectation is that if the model show no response, then the explanations should not.

    References:
        Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., Dähne Sven,
        Erhan Dumitru and Kim Been. "THE (UN)RELIABILITY OF SALIENCY METHODS" Article (2017).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", False)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.input_shift = self.kwargs.get("input_shift", -1)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("input shift 'input_shift'"),
                citation=(
                    "Kindermans Pieter-Jan, Hooker Sarah, Adebayo Julius, Alber Maximilian, Schütt Kristof T., "
                    "Dähne Sven, Erhan Dumitru and Kim Been. 'THE (UN)RELIABILITY OF SALIENCY METHODS' Article (2017)."
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None],
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
            args: optional args
            kwargs: optional dict

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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
        assert_attributions(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                print(
                    "An absolute operation on the attributions is skipped"
                    "since inconsistent results can be expected if applied."
                )

            if self.normalise:
                print(
                    "A normalising operation on the attributions is skipped"
                    "since inconsistent results can be expected if applied."
                )

            x_shifted = self.perturb_func(
                img=x.flatten(),
                **{
                    "indices": np.arange(0, len(x.flatten())),
                    "input_shift": self.input_shift,
                },
            )
            assert_perturbation_caused_change(x=x, x_perturbed=x_shifted)

            # Generate explanation based on shifted input x.
            a_shifted = explain_func(
                model=model, inputs=x_shifted, targets=y, **self.kwargs
            )

            if self.abs:
                print(
                    "An absolute operation on the attributions is skipped"
                    "since inconsistent results can be expected if applied."
                )

            if self.normalise:
                print(
                    "A normalising operation on the attributions is skipped"
                    "since inconsistent results can be expected if applied."
                )

            # Check if explanation of shifted input is similar to original.
            if (a.flatten() != a_shifted.flatten()).all():
                self.last_results.append(True)
            else:
                self.last_results.append(False)

        self.all_results.append(self.last_results)

        return self.last_results
