"""This module contains the collection of faithfulness metrics to evaluate attribution-based explanations of neural network models."""
from .base import Metric
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *
from ..helpers.normalise_func import *
from ..helpers.warn_func import *


class FaithfulnessCorrelation(Metric):
    """
    Implementation of faithfulness correlation by Bhatt et al., 2020.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating feature-based model
        explanations." arXiv preprint arXiv:2005.00631 (2020).

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
        self.nr_runs = self.kwargs.get("nr_runs", 100)
        self.subset_size = self.kwargs.get("subset_size", 224)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.return_aggregate = self.kwargs.get("return_aggregate", True)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', size of subset |S| 'subset_size'"
                    " and the number of runs (for each input and explanation pair) "
                    "'nr_runs'"
                ),
                citation=(
                    "Bhatt, Umang, Adrian Weller, and José MF Moura. 'Evaluating and aggregating "
                    "feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)"
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
            >> metric = FaithfulnessCorrelation(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            logit_deltas = []
            att_sums = []

            # For each test data point, execute a couple of runs.
            for i_ix in range(self.nr_runs):

                # Randomly mask by subset size.
                a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{
                        "indices": a_ix,
                        "perturb_baseline": self.perturb_baseline,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        model(
                            torch.Tensor(x_perturbed)
                            .reshape(
                                1,
                                self.nr_channels,
                                self.img_size,
                                self.img_size,
                            )
                            .to(self.kwargs.get("device", None))
                        )[:, y]
                    )

                logit_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions of the random subset.
                att_sums.append(np.sum(a[a_ix]))

            self.last_results.append(self.similarity_func(a=att_sums, b=logit_deltas))

        if self.return_aggregate:
            self.last_results = [np.mean(self.last_results)]
        else:
            self.last_results = self.last_results

        self.all_results.append(self.last_results)

        return self.last_results


class FaithfulnessEstimate(Metric):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).
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
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.img_size = self.kwargs.get("img_size", 224)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and similarity function "
                    "'similarity_func'"
                ),
                citation=(
                    "Alvarez-Melis, David, and Tommi S. Jaakkola. 'Towards robust interpretability"
                    " with self-explaining neural networks.' arXiv preprint arXiv:1806.07538 (2018)"
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_features_in_step(
            features_in_step=self.features_in_step, img_size=self.img_size
        )
        if self.max_steps_per_input is not None:
            assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )
            self.set_features_in_step = set_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )

    def __call__(
        self,
        model,
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
            >> metric = FaithfulnessEstimate(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{
                        "indices": a_ix,
                        "perturb_baseline": self.perturb_baseline,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        model(
                            torch.Tensor(x_perturbed)
                            .reshape(
                                1,
                                self.nr_channels,
                                self.img_size,
                                self.img_size,
                            )
                            .to(self.kwargs.get("device", None))
                        )[:, y]
                    )
                pred_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            self.last_results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        self.all_results.append(self.last_results)

        return self.last_results


class MonotonicityArya(Metric):
    """
    Implementation of Montonicity Metric by Arya at el., 2019.

    Montonicity tests if adding more positive evidence increases the probability
    of classification in the specified class.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Arya, Vijay, et al. "One explanation does not fit all: A toolkit and taxonomy of ai explainability
        techniques." arXiv preprint arXiv:1909.03012 (2019).
        2) Luss, Ronny, et al. "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.img_size = self.kwargs.get("img_size", 224)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Arya, Vijay, et al. 'One explanation does not fit all: A toolkit and taxonomy"
                    " of ai explainability techniques.' arXiv preprint arXiv:1909.03012 (2019)"
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_features_in_step(
            features_in_step=self.features_in_step, img_size=self.img_size
        )
        if self.max_steps_per_input is not None:
            assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )
            self.set_features_in_step = set_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )

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
            last_results: a list of bool(s) with the evaluation outcome of concerned batch

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
            >> metric = MonotonicityArya(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            preds = []

            baseline_value = get_baseline_value(
                choice=self.perturb_baseline, img=x, **kwargs
            )

            # Copy the input x but fill with baseline values.
            x_baseline = (
                torch.Tensor()
                .new_full(size=x.shape, fill_value=baseline_value)
                .numpy()
                .flatten()
            )

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_baseline = self.perturb_func(
                    img=x_baseline,
                    **{"indices": a_ix, "fixed_values": x.flatten()[a_ix]},
                )

                # Predict on perturbed input x (that was initially filled with a constant 'perturb_baseline' value).
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_baseline)
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

                preds.append(y_pred_perturb)

            self.last_results.append(np.all(np.diff(preds) >= 0))

        self.all_results.append(self.last_results)

        return self.last_results


class MonotonicityNguyen(Metric):
    """
    Implementation of Montonicity Metric by Nguyen at el., 2020.

    Monotonicity measures the (Spearman’s) correlation coefficient of the absolute values of the attributions
    and the uncertainty in probability estimation. The paper argues that if attributions are not monotonic
    then they are not providing the correct importance of the feature.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = Callable
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.eps = self.kwargs.get("eps", 1e-5)
        self.nr_samples = self.kwargs.get("nr_samples", 100)
        self.img_size = self.kwargs.get("img_size", 224)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', threshold value 'eps' and number "
                    "of samples to iterate over 'nr_samples'"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)"
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_features_in_step(
            features_in_step=self.features_in_step, img_size=self.img_size
        )
        if self.max_steps_per_input is not None:
            assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )
            self.set_features_in_step = set_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )

    def __call__(
        self,
        model,
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
            >> metric = MonotonicityNguyen(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Predict on input x.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
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

            inv_pred = 1.0 if np.abs(y_pred) < self.eps else 1.0 / np.abs(y_pred)
            inv_pred = inv_pred ** 2

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (ascending).
            a_indices = np.argsort(a)

            atts = []
            vars = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]

                y_pred_perturbs = []

                for n in range(self.nr_samples):

                    x_perturbed = self.perturb_func(
                        img=x.flatten(),
                        **{
                            "indices": a_ix,
                            "perturb_baseline": self.perturb_baseline,
                        },
                    )
                    assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                    # Predict on perturbed input x.
                    with torch.no_grad():
                        y_pred_perturb = float(
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
                    y_pred_perturbs.append(y_pred_perturb)

                vars.append(
                    float(
                        np.mean((np.array(y_pred_perturb) - np.array(y_pred)) ** 2)
                        * inv_pred
                    )
                )
                atts.append(float(sum(a[a_ix])))

            self.last_results.append(self.similarity_func(a=atts, b=vars))

        self.all_results.append(self.last_results)

        return self.last_results


class PixelFlipping(Metric):
    """
    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plot_pixel_flipping_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.img_size = self.kwargs.get("img_size", 224)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                    " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015) "
                    "e0130140."
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_features_in_step(
            features_in_step=self.features_in_step, img_size=self.img_size
        )
        if self.max_steps_per_input is not None:
            assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )
            self.set_features_in_step = set_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )

    def __call__(
        self,
        model,
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
            >> metric = PixelFlipping(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            preds = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x_perturbed,
                    **{
                        "indices": a_ix,
                        "perturb_baseline": self.perturb_baseline,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
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
                preds.append(y_pred_perturb)

            self.last_results.append(preds)

        self.all_results.append(self.last_results)

        return self.last_results


class RegionPerturbation(Metric):
    """

    Implementation of Region Perturbation by Samek et al., 2015.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
        neural network has learned." IEEE transactions on neural networks and
        learning systems 28.11 (2016): 2660-2673.

    Current assumptions:
        -Done according to Most Relevant First (MoRF) and Area Over the Perturbation Curve
        (AOPC).
        - 9 x 9 patch sizes was used in the paper as regions, but using 8 x 8
        to make sure non-overlapping
        - they called it "area over the MoRF perturbation curve" it
        looks like a simple deduction of function outputs?

    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plot_region_perturbation_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.regions_evaluation = self.kwargs.get("regions_evaluation", 100)
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.random_order = self.kwargs.get("random_order", False)
        self.order = self.kwargs.get("order", "MoRF").lower()
        self.img_size = self.kwargs.get("img_size", 224)
        self.text_warning = (
            "\nThe Region perturbation metric is likely to be sensitive to the choice of "
            "baseline value 'perturb_baseline', the patch size for masking 'patch_size' and number of regions to"
            " evaluate 'regions_evaluation'. "
            "\nGo over and select each hyperparameter of the metric carefully to avoid misinterpretation of scores. "
            "\nTo view all relevant hyperparameters call .get_params of the metric instance. "
            "\nFor further reading: Samek, Wojciech, et al. 'Evaluating the visualization of what a "
            "deep neural network has learned.' IEEE transactions on neural networks and learning "
            "systems 28.11 (2016): 2660-2673.' PloS one 10.7 (2015): e0130140."
        )
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        assert_patch_size(patch_size=self.patch_size, img_size=self.img_size)
        assert_attributions_order(order=self.order)
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=(
                    "Bach, Sebastian, et al. 'On pixel-wise explanations for non-linear classifier"
                    " decisions by layer - wise relevance propagation.' PloS one 10.7 (2015): "
                    "e0130140"
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
            args: optional args
            kwargs: optional dict

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
            >> metric = RegionPerturbation(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = {k: None for k in range(len(x_batch))}

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

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.nn.Softmax()(
                            torch.Tensor(x)
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

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            patches = []
            sub_results = []
            x_perturbed = x.copy()

            att_sums = np.zeros(
                (
                    int(a.shape[0] / self.patch_size),
                    int(a.shape[1] / self.patch_size),
                )
            )

            # Get patch indices of sorted attributions (descending).
            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):
                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[
                        top_left_x : top_left_x + self.patch_size,
                        top_left_y : top_left_y + self.patch_size,
                    ].sum()
                    patches.append([top_left_y, top_left_x])

            if self.order == "morf":

                # Order attributions according to the most relevant first.
                patch_order = {
                    k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)
                }

            else:

                # Order attributions according to the least relevant first.
                patch_order = {
                    k: v for k, v in zip(np.argsort(att_sums, axis=None), patches)
                }

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(min(self.regions_evaluation, len(patch_order))):

                # Calculate predictions on a random order.
                if self.random_order:
                    order = random.randint(0, len(patch_order))
                    top_left_y = patch_order[order][0]
                    top_left_x = patch_order[order][1]
                else:
                    top_left_y = patch_order[k][0]
                    top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(
                    x_perturbed,
                    **{
                        "patch_size": self.patch_size,
                        "nr_channels": self.nr_channels,
                        "perturb_baseline": self.perturb_baseline,
                        "top_left_y": top_left_y,
                        "top_left_x": top_left_x,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_perturb = float(
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

                sub_results.append(y_pred - y_pred_perturb)

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results


class Selectivity(Metric):
    """
    Implementation of Selectivity test by Montavan et al., 2018.

    At each iteration, a patch of size 4 x 4 corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plot_selectivity_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.img_size = self.kwargs.get("img_size", 224)
        self.last_results = {}
        self.all_results = []

        # Asserts and warnings.
        assert_patch_size(patch_size=self.patch_size, img_size=self.img_size)
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the patch size for masking"
                    " 'patch_size'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73 (2018): 1-15"
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
            args: optional args
            kwargs: optional dict

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
            >> metric = Selectivity(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = {k: None for k in range(len(x_batch))}

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

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
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

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            patches = []
            sub_results = []
            x_perturbed = x.copy()

            att_sums = np.zeros(
                (
                    int(a.shape[0] / self.patch_size),
                    int(a.shape[1] / self.patch_size),
                )
            )

            # Get patch indices of sorted attributions (descending).
            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):

                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[
                        top_left_x : top_left_x + self.patch_size,
                        top_left_y : top_left_y + self.patch_size,
                    ].sum()
                    patches.append([top_left_y, top_left_x])

            patch_order = {
                k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)
            }

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(len(patch_order)):
                top_left_y = patch_order[k][0]
                top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(
                    x_perturbed,
                    **{
                        "patch_size": self.patch_size,
                        "nr_channels": self.nr_channels,
                        "perturb_baseline": self.perturb_baseline,
                        "top_left_y": top_left_y,
                        "top_left_x": top_left_x,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_perturb = float(
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

                sub_results.append(y_pred_perturb)

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results


class SensitivityN(Metric):
    """
    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is reported. Sampling is performed using a uniform probability distribution over the features.

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         - Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.
         - I don't get why they have so high correlation in the paper, maybe using a better baseline_value?
         - Also I don't get why correlation is only reported positive?

    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_negative)
        self.default_plot_func = plot_sensitivity_n_experiment
        self.disable_warnings = self.kwargs.get("disable_warnings", False)
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.n_max_percentage = self.kwargs.get("n_max_percentage", 0.8)
        self.img_size = self.kwargs.get("img_size", 224)
        self.features_in_step = self.kwargs.get("features_in_step", 1)
        self.max_features = int(
            (0.8 * self.img_size * self.img_size) // self.features_in_step
        )
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the patch size for masking "
                    "'patch_size', similarity function 'similarity_func' and the number "
                    "of features to iteratively evaluate 'n_max_percentage'"
                ),
                citation=(
                    "Ancona, Marco, et al. 'Towards better understanding of gradient-based "
                    "attribution methods for deep neural networks.' arXiv preprint "
                    "arXiv:1711.06104 (2017)"
                ),
            )
            warn_attributions(normalise=self.normalise, abs=self.abs)
        assert_features_in_step(
            features_in_step=self.features_in_step, img_size=self.img_size
        )
        if self.max_steps_per_input is not None:
            assert_max_steps(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )
            self.set_features_in_step = set_features_in_step(
                max_steps_per_input=self.max_steps_per_input,
                img_size=self.img_size,
            )

    def __call__(
        self,
        model,
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
            >> metric = SensitivityN(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        sub_results_pred_deltas = {k: [] for k in range(len(x_batch))}
        sub_results_att_sums = {k: [] for k in range(len(x_batch))}

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            a = a.flatten()

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on x.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
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

            att_sums = []
            pred_deltas = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                if i_ix <= self.max_features:

                    # Perturb input by indices of attributions.
                    a_ix = a_indices[
                        (self.features_in_step * i_ix) : (
                            self.features_in_step * (i_ix + 1)
                        )
                    ]
                    x_perturbed = self.perturb_func(
                        img=x_perturbed,
                        **{
                            "indices": a_ix,
                            "perturb_baseline": self.perturb_baseline,
                        },
                    )
                    assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                    # Sum attributions.
                    att_sums.append(float(a[a_ix].sum()))

                    with torch.no_grad():
                        y_pred_perturb = float(
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
                    pred_deltas.append(y_pred - y_pred_perturb)

            sub_results_att_sums[sample] = att_sums
            sub_results_pred_deltas[sample] = pred_deltas

        # Re-arrange sublists so that they are sorted by n.
        sub_results_pred_deltas_l = {k: [] for k in range(self.max_features)}
        sub_results_att_sums_l = {k: [] for k in range(self.max_features)}

        for k in range(self.max_features):
            for sublist1 in list(sub_results_pred_deltas.values()):
                sub_results_pred_deltas_l[k].append(sublist1[k])
            for sublist2 in list(sub_results_att_sums.values()):
                sub_results_att_sums_l[k].append(sublist2[k])

        # Measure similarity for each n.
        self.last_result = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(self.max_features)
        ]
        self.all_results.append(self.last_result)

        return self.last_result


class IterativeRemovalOfFeatures(Metric):
    """
    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    The metric computes the area over the curve per class for sorted mean importances
    of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected),
    averaged over several test samples.

    References:
        1) Rieger, Laura, and Lars Kai Hansen. "Irof: a low resource evaluation metric for
        explanation methods." arXiv preprint arXiv:2003.08747 (2020).

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
        self.segmentation_method = self.kwargs.get("segmentation_method", "slic")
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "mean")
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
                    "baseline value 'perturb_baseline' and the method to segment "
                    "the image 'segmentation_method' (including all its associated hyperparameters)"
                ),
                citation=(
                    "Rieger, Laura, and Lars Kai Hansen. 'Irof: a low resource evaluation metric "
                    "for explanation methods.' arXiv preprint arXiv:2003.08747 (2020)"
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
            >> metric = IROF(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency, **{}}
        """

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            # Predict on x.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
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

            # Segment image.
            segments = get_superpixel_segments(
                img=np.moveaxis(x, 0, -1).astype("double"),
                **kwargs,
            )
            nr_segments = segments.max()
            assert_nr_segments(nr_segments=nr_segments)

            # Calculate average attribution of each segment.
            att_segs = np.zeros(nr_segments)
            for i, s in enumerate(range(nr_segments)):
                att_segs[i] = np.mean(a[segments == s])

            # Sort segments based on the mean attribution (descending order).
            s_indices = np.argsort(-att_segs)

            preds = []

            for i_ix, s_ix in enumerate(s_indices):

                # Perturb input by indices of attributions.
                a_ix = np.nonzero(
                    np.repeat((segments == s_ix).flatten(), self.nr_channels)
                )[0]

                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{
                        "indices": a_ix,
                        "perturb_baseline": self.perturb_baseline,
                    },
                )
                assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
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

                # Normalise the scores to be within [0, 1].
                preds.append(float(y_pred_perturb / y_pred))

            # self.last_results.append(1-auc(preds, np.arange(0, len(preds))))
            self.last_results.append(np.trapz(np.array(preds), dx=1.0))

        self.last_results = [np.mean(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """Calculate the area over the curve (AOC) score for several test samples."""
        return [np.mean(results) for results in self.all_results]
