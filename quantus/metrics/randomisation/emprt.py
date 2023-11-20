"""This module contains the implementation of the enhanced Model Parameter Randomisation Test metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Collection,
    Iterable,
)
import os
import numpy as np
from tqdm.auto import tqdm
import torch

from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max

# from quantus.functions import complexity_func
from quantus.metrics.base import Metric
from quantus.helpers.enums import (
    ModelType,
    DataType,
    ScoreDirection,
    EvaluationCategory,
)


class eMPRT(Metric):
    """
    Implementation of the NAME by AUTHOR et. al., 2023.

    INSERT DESC.

    References:
        1) INSERT SOURCE

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Enhanced Model Parameter Randomisation Test"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.RANDOMISATION

    def __init__(
        self,
        complexity_func: Optional[Callable] = None,
        complexity_func_kwargs: Optional[dict] = None,
        layer_order: str = "bottom_up",
        nr_samples: Optional[int] = None,
        seed: int = 42,
        compute_delta: bool = True,
        compute_rate_of_change: bool = True,
        compute_delta_explanation_vs_model: bool = True,
        compute_correlation: bool = True,
        compute_last_complexity: bool = True,
        return_delta_explanation_vs_model: bool = False,
        return_fraction: bool = False,
        return_rate_of_change: bool = True,
        return_average_sample_score: bool = False,
        return_correlation: bool = False,
        return_last_complexity: bool = False,
        return_delta_explanation: bool = False,
        skip_layers: bool = False,
        similarity_func: Optional[Callable] = None,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: callable
            Similarity function applied to compare input and perturbed input, default=correlation_spearman.
        layer_order: string
            Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        seed: integer
            Seed used for the random generator, default=42.
        return_average_sample_score: boolean
            Indicates whether return one float per sample, representing the average
            correlation coefficient across the layers for that sample.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=True.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        if normalise_func is None:
            normalise_func = normalise_by_max

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Set seed for reproducibility.
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)

            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.enabled = False

        # Save metric-specific attributes.
        if complexity_func is None:
            complexity_func = discrete_entropy

        if complexity_func_kwargs is None:
            complexity_func_kwargs = {}

        if similarity_func is None:
            similarity_func = similarity_func.correlation_spearman

        self.complexity_func = complexity_func
        self.complexity_func_kwargs = complexity_func_kwargs
        self.similarity_func = similarity_func
        self.layer_order = layer_order
        self.nr_samples = nr_samples
        self.compute_delta = compute_delta
        self.compute_rate_of_change = compute_rate_of_change
        self.compute_delta_explanation_vs_model = compute_delta_explanation_vs_model
        self.compute_correlation = compute_correlation
        self.compute_last_complexity = compute_last_complexity
        self.return_average_sample_score = return_average_sample_score
        self.return_fraction = return_fraction
        self.return_rate_of_change = return_rate_of_change
        self.return_delta_explanation_vs_model = return_delta_explanation_vs_model
        self.return_correlation = return_correlation
        self.return_last_complexity = return_last_complexity
        self.return_delta_explanation = return_delta_explanation
        self.skip_layers = skip_layers

        # Asserts and warnings.
        assert (
            sum(
                [
                    self.return_fraction,
                    self.return_average_sample_score,
                    self.return_correlation,
                    self.return_last_complexity,
                    self.return_delta_explanation,
                    self.return_delta_explanation_vs_model,
                    self.return_rate_of_change,
                ]
            )
            == 1
        ), "Set one of the possible 'return' arguments to True."

        asserts.assert_layer_order(layer_order=self.layer_order)
        if not self.disable_warnings:
            warn.warn_parameterisation(
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
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        attributions_path: str = None,
        **kwargs,
    ) -> Union[List[float], float, Dict[str, List[float]], Collection[Any]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        The content of evaluation_scores will be appended to all_evaluation_scores (list) at the end of
        the evaluation call.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

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
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """

        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        data = self.general_preprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
        )

        # Get model and data.
        model = data["model"]
        x_batch = data["x_batch"]
        y_batch = data["y_batch"]
        a_batch = data["a_batch"]

        # Get number of iterations from number of layers.
        n_layers = len(list(model.get_random_layer_generator(order=self.layer_order)))
        model_iterator = tqdm(
            model.get_random_layer_generator(order=self.layer_order),
            total=n_layers,
            disable=not self.display_progressbar,
        )

        # Get the number of bins for discrete entropy calculation.
        if "n_bins" not in self.complexity_func_kwargs:
            self.find_n_bins(
                a_batch=a_batch,
                n_bins_default=self.complexity_func_kwargs.get("n_bins_default", 100),
                min_n_bins=self.complexity_func_kwargs.get("min_n_bins", 10),
                max_n_bins=self.complexity_func_kwargs.get("max_n_bins", 200),
                debug=self.complexity_func_kwargs.get("debug", False),
            )

        # Compute the explanation_scores given uniformly sampled explanation.
        if self.nr_samples is None:
            self.nr_samples = len(a_batch)

        # Initialise arrays.
        self.delta_explanation_scores = np.zeros((self.nr_samples))
        self.delta_model_scores = np.zeros((self.nr_samples))
        self.fraction_explanation_scores = np.zeros((self.nr_samples))
        self.fraction_model_scores = np.zeros((self.nr_samples))
        self.delta_explanation_vs_models = np.zeros((self.nr_samples))
        self.correlation_scores = np.zeros((self.nr_samples))
        self.rate_of_change_scores = np.zeros((self.nr_samples))
        self.explanation_scores = {}
        self.model_scores = {}

        for l_ix, (layer_name, random_layer_model) in enumerate(model_iterator):

            if l_ix == 0:

                # Generate an explanation with perturbed model.
                a_batch_original = self.explain_func(
                    model=model.get_model(),
                    inputs=x_batch,
                    targets=y_batch,
                    **self.explain_func_kwargs,
                )

                self.explanation_scores["orig"] = []
                for a_ix, a_ori in enumerate(a_batch_original):
                    score = self.evaluate_instance(
                        model=model,
                        x=x_batch[0],
                        y=None,
                        s=None,
                        a=a_ori,
                    )
                    self.explanation_scores["orig"].append(score)

                # Compute entropy of the output layer.
                self.model_scores["orig"] = []
                for y_ix, y_pred in enumerate(model.predict(x_batch)):
                    score = entropy(a=y_pred, x=y_pred)
                    self.model_scores["orig"].append(score)

            # Skip layers if computing delta.
            if (
                self.skip_layers
                and self.compute_delta
                and (l_ix + 1) < len(model_iterator)
            ):
                continue

            # Score explanation complexity.
            explanation_scores = []

            # Generate an explanation with perturbed model.
            a_batch_perturbed = self.explain_func(
                model=random_layer_model,
                inputs=x_batch,
                targets=y_batch,
                **self.explain_func_kwargs,
            )

            # Get id for storing data.
            if attributions_path is not None:
                savepath = os.path.join(attributions_path, f"{l_ix}-{layer_name}")
                os.makedirs(savepath, exist_ok=True)
                last_id = 0
                for fname in os.listdir(savepath):
                    if "original_attribution_" in fname:
                        id = (
                            int(fname.split("original_attribution_")[1].split(".")[0])
                            > last_id
                        )
                        if id > last_id:
                            last_id = id

            batch_iterator = enumerate(zip(a_batch, a_batch_perturbed))
            for instance_id, (a_ix, a_perturbed) in batch_iterator:
                score = self.evaluate_instance(
                    model=random_layer_model,
                    x=x_batch[0],
                    y=None,
                    s=None,
                    a=a_perturbed,
                )
                explanation_scores.append(score)

                # Save data.
                if attributions_path is not None:
                    np.save(
                        os.path.join(savepath, f"input_{last_id+instance_id}.npy"),
                        x_batch[instance_id],
                    )
                    np.save(
                        os.path.join(
                            savepath, f"original_attribution_{last_id+instance_id}.npy"
                        ),
                        a_ix,
                    )
                    np.save(
                        os.path.join(
                            savepath, f"perturbed_attribution_{last_id+instance_id}.npy"
                        ),
                        a_perturbed,
                    )

            # Score the model complexity.
            model_scores = []

            # Wrap the model.
            random_layer_model_wrapped = utils.get_wrapped_model(
                model=random_layer_model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

            # Predict and save scores.
            y_preds = random_layer_model_wrapped.predict(x_batch)
            for y_ix, y_pred in enumerate(y_preds):
                score = entropy(a=y_pred, x=y_pred)
                model_scores.append(score)

            # Save explanation_scores scores in a result dictionary.
            self.explanation_scores[layer_name] = explanation_scores
            self.model_scores[layer_name] = model_scores

        # Call post-processing.
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        # If compute correlation score (model and explanations)
        if self.compute_correlation:
            self.correlation_scores = (
                self.recompute_model_explanation_correlation_per_sample()
            )

        # If compute the last complexity score.
        if self.compute_last_complexity:
            self.last_complexity_scores = self.recompute_last_correlation_per_sample()

        # If compute delta score per sample (model and explanations).
        if self.compute_delta:

            # Compute deltas for explanation scores.
            scores = list(self.explanation_scores.values())
            self.delta_explanation_scores = [
                b - a for a, b in zip(scores[0], scores[-1])
            ]

            # Compute deltas for model scores.
            scores = list(self.model_scores.values())
            self.delta_model_scores = [b - a for a, b in zip(scores[0], scores[-1])]

            # Compute fraction for explanation scores.
            scores = list(self.explanation_scores.values())
            self.fraction_explanation_scores = [
                b / a if a != 0 else np.nan for a, b in zip(scores[0], scores[-1])
            ]  # eMPRT original!

            # Compute fraction for explanation scores.
            scores = list(self.model_scores.values())
            self.fraction_model_scores = [
                b / a if a != 0 else np.nan for a, b in zip(scores[0], scores[-1])
            ]

        # If compute delta skill score per sample (model and explanations).
        if self.compute_delta_explanation_vs_model:
            self.delta_explanation_vs_models = [
                b / a if a != 0 else np.nan
                for a, b in zip(
                    self.fraction_model_scores, self.fraction_explanation_scores
                )
            ]

        # If compute delta skill score per sample (model and explanations).
        if self.compute_rate_of_change:
            scores = list(self.explanation_scores.values())
            self.rate_of_change_scores = [
                (b - a) / a for a, b in zip(scores[0], scores[-1])
            ]

        # If return one score per sample.
        if self.return_average_sample_score:
            self.evaluation_scores = self.recompute_average_complexity_per_sample()

        # If return delta score per sample.
        if self.return_fraction:
            self.evaluation_scores = self.fraction_explanation_scores

        # If return delta score per sample.
        if self.return_delta_explanation_vs_model:
            self.evaluation_scores = self.delta_explanation_vs_models

        # If return delta score per sample.
        if self.return_correlation:
            self.evaluation_scores = self.correlation_scores

        if self.return_last_complexity:
            self.evaluation_scores = self.last_complexity_scores

        if self.return_rate_of_change:
            self.evaluation_scores = self.rate_of_change_scores

        # If return one aggregate score for all samples.
        if self.return_aggregate:
            self.evaluation_scores = [self.aggregate_func(self.evaluation_scores)]

        # Return all_evaluation_scores according to Quantus.
        self.all_evaluation_scores.append(self.evaluation_scores)

        return self.evaluation_scores

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        i: integer
            The evaluation instance.
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """
        if self.normalise:
            a = self.normalise_func(a, **self.normalise_func_kwargs)

        if self.abs:
            a = np.abs(a)

        # Compute distance measure.
        return self.complexity_func(a=a, x=x, **self.complexity_func_kwargs)

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        # Additional explain_func assert, as the one in general_preprocess()
        # won't be executed when a_batch != None.

        asserts.assert_explain_func(explain_func=self.explain_func)

    def recompute_model_explanation_correlation_per_sample(
        self,
    ) -> Union[List[List[Any]], Dict[int, List[Any]]]:

        assert isinstance(self.explanation_scores, dict), (
            "To compute the correlation between model and explanation per sample for "
            "enhanced Model Parameter Randomisation Test, 'explanation_scores' "
            "must be of type dict."
        )
        layer_length = len(
            self.explanation_scores[list(self.explanation_scores.keys())[0]]
        )
        explanation_scores: Dict[int, list] = {
            sample: [] for sample in range(layer_length)
        }
        model_scores: Dict[int, list] = {sample: [] for sample in range(layer_length)}

        for sample in explanation_scores.keys():
            for layer in self.explanation_scores:
                explanation_scores[sample].append(
                    float(self.explanation_scores[layer][sample])
                )
                model_scores[sample].append(float(self.model_scores[layer][sample]))

        corr_coeffs = []
        for sample in explanation_scores.keys():
            corr_coeffs.append(
                self.similarity_func(model_scores[sample], explanation_scores[sample])
            )

        return corr_coeffs

    def recompute_average_complexity_per_sample(
        self,
    ) -> Union[List[List[Any]], Dict[int, List[Any]]]:

        assert isinstance(self.explanation_scores, dict), (
            "To compute the average correlation coefficient per sample for "
            "enhanced Model Parameter Randomisation Test, 'explanation_scores' "
            "must be of type dict."
        )
        layer_length = len(
            self.explanation_scores[list(self.explanation_scores.keys())[0]]
        )
        results: Dict[int, list] = {sample: [] for sample in range(layer_length)}

        for sample in results:
            for layer in self.explanation_scores:
                if layer == "orig":
                    continue
                results[sample].append(float(self.explanation_scores[layer][sample]))
            results[sample] = np.mean(results[sample])

        corr_coeffs = list(results.values())

        return corr_coeffs

    def recompute_last_correlation_per_sample(
        self,
    ) -> Union[List[List[Any]], Dict[int, List[Any]]]:

        assert isinstance(self.explanation_scores, dict), (
            "To compute the last correlation coefficient per sample for "
            "Model Parameter Randomisation Test, 'explanation_scores' "
            "must be of type dict."
        )
        corr_coeffs = list(self.explanation_scores.values())[-1]

        return corr_coeffs

    def find_n_bins(
        self,
        a_batch: np.array,
        n_bins_default: int = 100,
        min_n_bins: int = 10,
        max_n_bins: int = 200,
        debug: bool = True,
    ) -> None:

        if self.normalise:
            a_batch = self.normalise_func(a, **self.normalise_func_kwargs)
        if self.abs:
            a_batch = np.abs(a_batch)

        rule_name = self.complexity_func_kwargs.get("rule", None)
        rule = RULES_N_BINS.get(rule_name)

        if debug:
            print(f"\tMax and min value of a_batch=({a_batch.min()}, {a_batch.max()})")

        if not rule:
            self.complexity_func_kwargs["n_bins"] = n_bins_default
            if debug:
                print(f"\tNo rule found, 'n_bins' set to 100.")
            return None

        n_bins = rule(a_batch=a_batch)
        n_bins = max(min(n_bins, max_n_bins), min_n_bins)
        self.complexity_func_kwargs["n_bins"] = n_bins

        if debug:
            print(
                f"\tRule '{rule_name}' -> n_bins={n_bins} but with min={min_n_bins} and max={max_n_bins}, 'n_bins' set to {self.complexity_func_kwargs['n_bins']}."
            )
