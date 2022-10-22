"""This module contains the implementation of the Model Parameter Sensitivity metric."""

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
import numpy as np
from tqdm.auto import tqdm

from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.similarity_func import correlation_spearman
from quantus.metrics.base import Metric


class ModelParameterRandomisation(Metric):
    """
    Implementation of the Model Parameter Randomization Method by Adebayo et. al., 2018.

    The Model Parameter Randomization measures the distance between the original attribution and a newly computed
    attribution throughout the process of cascadingly/independently randomizing the model parameters of one layer
    at a time.

    Assumptions:
        - In the original paper multiple distance measures are taken: Spearman rank correlation (with and without abs),
        HOG and SSIM. We have set Spearman as the default value.

    References:
        1) Julius Adebayo et al.: "Sanity Checks for Saliency Maps."
        NeurIPS (2018): 9525-9536.
    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Callable = None,
        layer_order: str = "independent",
        seed: int = 42,
        return_sample_correlation: bool = False,
        abs: bool = True,
        normalise: bool = True,
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
            Similarity function applied to compare input and perturbed input,
            default=correlation_spearman.
        layer_order: string
            Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        seed: integer
            Seed used for the random generator, default=42.
        return_sample_correlation: boolean
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

        # Save metric-specific attributes.
        if similarity_func is None:
            similarity_func = correlation_spearman
        self.similarity_func = similarity_func
        self.layer_order = layer_order
        self.seed = seed
        self.return_sample_correlation = return_sample_correlation

        # Results are returned/saved as a dictionary not like in the super-class as a list.
        self.last_results = {}

        # Asserts and warnings.
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
        **kwargs,
    ) -> Union[List[float], float, Dict[str, List[float]], Collection[Any]]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        The content of last_results will be appended to all_results (list) at the end of
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
        last_results: list
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
        model = data["model"]
        x_batch = data["x_batch"]
        y_batch = data["y_batch"]
        a_batch = data["a_batch"]

        # Results are returned/saved as a dictionary not as a list as in the super-class.
        self.last_results = {}

        # Get number of iterations from number of layers.
        n_layers = len(list(model.get_random_layer_generator(order=self.layer_order)))

        model_iterator = tqdm(
            model.get_random_layer_generator(order=self.layer_order, seed=self.seed),
            total=n_layers,
            disable=not self.display_progressbar,
        )

        for layer_name, random_layer_model in model_iterator:

            similarity_scores = [None for _ in x_batch]

            # Generate an explanation with perturbed model.
            a_batch_perturbed = self.explain_func(
                model=random_layer_model,
                inputs=x_batch,
                targets=y_batch,
                **self.explain_func_kwargs,
            )

            batch_iterator = enumerate(zip(a_batch, a_batch_perturbed))
            for instance_id, (a_instance, a_instance_perturbed) in batch_iterator:
                result = self.evaluate_instance(
                    model=random_layer_model,
                    x=None,
                    y=None,
                    s=None,
                    a=a_instance,
                    a_perturbed=a_instance_perturbed,
                )
                similarity_scores[instance_id] = result

            # Save similarity scores in a result dictionary.
            self.last_results[layer_name] = similarity_scores

        # Call post-processing
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        if self.return_sample_correlation:
            self.last_results = self.compute_correlation_per_sample()

        if self.return_aggregate:
            assert self.return_sample_correlation, (
                "You must set 'return_average_correlation_per_sample'"
                " to True in order to compute te aggregat"
            )
            self.last_results = [self.aggregate_func(self.last_results)]

        self.all_results.append(self.last_results)

        return self.last_results

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        a: Optional[np.ndarray],
        s: Optional[np.ndarray],
        a_perturbed: Optional[np.ndarray] = None,
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
        a_perturbed: np.ndarray
            The perturbed attributions.

        Returns
        -------
        float
            The evaluation results.
        """
        if self.normalise:
            a_perturbed = self.normalise_func(a_perturbed, **self.normalise_func_kwargs)

        if self.abs:
            a_perturbed = np.abs(a_perturbed)

        # Compute distance measure.
        return self.similarity_func(a_perturbed.flatten(), a.flatten())

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

    def compute_correlation_per_sample(
        self,
    ) -> Union[List[List[Any]], Dict[int, List[Any]]]:

        assert isinstance(self.last_results, dict), (
            "To compute the average correlation coefficient per sample for "
            "Model Parameter Randomisation Test, 'last_result' "
            "must be of type dict."
        )
        layer_length = len(self.last_results[list(self.last_results.keys())[0]])
        results: Dict[int, list] = {sample: [] for sample in range(layer_length)}

        for sample in results:
            for layer in self.last_results:
                results[sample].append(float(self.last_results[layer][sample]))
            results[sample] = np.mean(results[sample])

        corr_coeffs = list(results.values())

        return corr_coeffs
