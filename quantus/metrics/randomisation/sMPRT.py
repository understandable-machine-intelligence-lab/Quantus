"""This module contains the implementation of the Model Parameter Randomisation with Sampling metric."""

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
import torch
from tqdm.auto import tqdm
from copy import deepcopy

from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.similarity_func import correlation_spearman
from quantus.metrics.randomisation.model_parameter_randomisation import ModelParameterRandomisation
from quantus.helpers.enums import (
    ModelType,
    DataType,
    ScoreDirection,
    EvaluationCategory,
)


class ModelParameterRandomisationSampling(ModelParameterRandomisation):
    """
    
    """

    name = "Model Parameter Randomisation Sampling"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.RANDOMISATION

    def __init__(
        self,
        n_noisy_models: int = 3,
        ng_std_level: Optional[float] = None,
        n_random_models: int = 1,
        similarity_func: Callable = None,
        layer_order: str = "independent",
        seed: int = None,
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
        num_draws: integer
            Number of randomization draws per layer.
            Higher numbers reduce noise in explanation functions but take longer to compute, default=10
        similarity_func: callable
            Similarity function applied to compare input and perturbed input, default=correlation_spearman.
        layer_order: string
            Indicated whether the model is randomized cascadingly or independently.
            Set order=top_down for cascading randomization, set order=independent for independent randomization,
            default="independent".
        seeds: List of integers
            Seeds used for the random generators, with as many seeds as num_draws, default=None.
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

        self.n_noisy_models = n_noisy_models
        self.ng_std_level = ng_std_level
        self.n_random_models = n_random_models

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)

            #torch.backends.cudnn.benchmark = False
            #torch.backends.cudnn.deterministic = True
            #torch.backends.cudnn.enabled = False

        super().__init__(
            similarity_func=similarity_func,
            layer_order=layer_order,
            return_sample_correlation=return_sample_correlation,
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
        attributions_path: str
            Optional path to store attributions as .npy files. default=None
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

        if self.ng_std_level is None:
            try:
                self.compute_noise_level_model(model=model, x_batch=x_batch, y_batch=y_batch)
            except:
                self.ng_std_level = 0.1
                print(
                    f"Unable to compute the noise level algorithmically based on the heuristic from the original"
                    f" paper by Bykov et al., (2021). Defaulting to a ng_std_level "
                    f"of {self.ng_std_level}.")

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
        self.evaluation_scores = {}

        # Get randomisable layers.
        randomisable_layers = model.get_randomisable_layer_names(order=self.layer_order)

        with tqdm(total=len(randomisable_layers)*self.n_noisy_models*self.n_random_models, disable=not self.display_progressbar) as pbar:
            for randomisation_id in range(self.n_random_models):
                for l, layer_name in enumerate(randomisable_layers):
                    similarity_scores = [None for _ in x_batch]
                    a_batch_perturbed_draws = []
                    _, random_layer_model = model.get_random_model_until_layer(order=self.layer_order, layer_name=layer_name)

                    # Generate self.num_draws perturbed models and explanations for each layer
                    for perturb in range(self.n_noisy_models):

                        # model_iterator = model_iterators[draw]
                        # layer_name, random_layer_model = next(x for x in model_iterator)
                        random_layer_model_draw = deepcopy(random_layer_model)

                        # Keep last perturbation as the original
                        if perturb != self.n_noisy_models - 1:
                            model.perturb_model_weights(random_layer_model_draw, std=self.ng_std_level)

                        # Generate an explanation with perturbed model.
                        a_batch_perturbed_draws.append(self.explain_func(
                            model=random_layer_model_draw,
                            inputs=x_batch,
                            targets=y_batch,
                            **self.explain_func_kwargs,
                        ))
                    
                    a_batch_perturbed = np.mean(a_batch_perturbed_draws, axis=0)

                    # Get id for storage
                    if attributions_path is not None and randomisation_id == self.n_random_models-1:
                        savepath = os.path.join(attributions_path, f"{l}-{layer_name}")
                        os.makedirs(savepath, exist_ok=True)
                        last_id = 0
                        for fname in os.listdir(savepath):
                            if "original_attribution_" in fname:
                                id = int(fname.split("original_attribution_")[1].split(".")[0]) > last_id
                                if id > last_id:
                                    last_id = id

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

                        if attributions_path is not None:
                            np.save(os.path.join(savepath, f"input_{last_id+instance_id}.npy"), x_batch[instance_id])
                            np.save(os.path.join(savepath, f"original_attribution_{last_id+instance_id}.npy"), a_instance)
                            np.save(os.path.join(savepath, f"perturbed_attribution_{last_id+instance_id}.npy"), a_instance_perturbed)

                    # Save similarity scores in a result dictionary.
                    if layer_name not in self.evaluation_scores.keys():
                        self.evaluation_scores[layer_name] = []
                    self.evaluation_scores[layer_name].append(similarity_scores)

                    pbar.update(1)

        # Call post-processing.
        self.custom_postprocess(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
        )

        if self.return_sample_correlation:
            self.evaluation_scores = self.compute_correlation_per_sample()

        if self.return_aggregate:
            assert self.return_sample_correlation, (
                "You must set 'return_average_correlation_per_sample'"
                " to True in order to compute te aggregat"
            )
            self.evaluation_scores = [self.aggregate_func(self.evaluation_scores)]

        self.all_evaluation_scores.append(self.evaluation_scores)

        return self.evaluation_scores

    def compute_noise_level_model(self, model, x_batch: np.array, y_batch: np.array):
        """
        Compute the noise level for a given model and input batch that results in a 5% accuracy drop.

        Parameters
        ----------
        self: object
            An instance of the current class.
        model: object
            A trained classification model.
        x_batch: np.ndarray
            A numpy array of input data samples of shape `(batch_size, channels, height, width)`.
        y_batch: np.ndarray
            A numpy array of integer labels of shape `(batch_size,)`.

        Returns
        -------
        std_drop : float
            The noise level (std) that results in a 5% accuracy drop.
        """

        # Wrap model into ModelInterface of Quantus.
        model_original = utils.get_wrapped_model(
            model=model,
            channel_first=self.channel_first,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
        )

        # Compute predictions of the original model
        preds_original = np.argmax(model_original.predict(x_batch), axis=1)
        acc_original = np.mean(np.equal(y_batch.astype(int), preds_original.astype(int)).astype(int))

        # Target accuracy after a 5% drop
        acc_target = acc_original - 0.05

        std = 0.01
        std_drop = None

        while std_drop is None:

            # Perturb model.
            model_perturbed = model_original.sample(
                mean=1.0, std=std, noise_type="multiplicative"
            )

            # Wrap model into ModelInterface of Quantus.
            model_perturbed = utils.get_wrapped_model(
                model=model_perturbed,
                channel_first=self.channel_first,
                softmax=self.softmax,
                device=self.device,
                model_predict_kwargs=self.model_predict_kwargs,
            )

            # Predict with the perturbed model.
            preds_perturbed = np.argmax(model_perturbed.predict(x_batch), axis=1)

            # Calculate fraction of similar predictions.
            acc_perturbed = np.mean(np.equal(preds_original.astype(int), preds_perturbed.astype(int)).astype(int))

            # If the accuracy drops by 5%, set std_drop and break.
            if acc_perturbed <= acc_target:
                std_drop = std
                break
            else:
                std += 0.01

        if self.debug:
            print(f"The current model experiences a 5% drop in accuracy with a std of {std_drop:.4f}.")

        return std_drop