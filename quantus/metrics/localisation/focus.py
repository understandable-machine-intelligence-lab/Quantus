"""This module contains the implementation of the AUC metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from quantus.helpers import asserts
from quantus.helpers import plotting
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max
from quantus.metrics.base import Metric


class Focus(Metric):
    """
    Implementation of Focus evaluation strategy by Arias et. al. 2022

    The Focus is computed through mosaics of instances from different classes, and the explanations these generate.
    Each mosaic contains four images: two images belonging to the target class (the specific class the feature
    attribution method is expected to explain) and the other two are chosen randomly from the rest of classes.
    Thus, the Focus estimates the reliability of feature attribution method’s output as the probability of the sampled
    pixels lying on an image of the target class of the mosaic. This is equivalent to the proportion
    of positive relevance lying on those images.

    References:
        1) Anna Arias-Duart et al.: "Focus! Rating XAI Methods
        and Finding Biases" FUZZ-IEEE (2022): 1-8.
    """

    @asserts.attributes_check
    def __init__(
        self,
        mosaic_shape: Optional[Any] = None,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = plotting.plot_focus,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
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

        # Save metric-specific attributes.
        self.mosaic_shape = mosaic_shape

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

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "no parameter. No parameters means nothing to be sensitive on. "
                    "Note, however, that Focus only works with image data and "
                    "a 'p_batch' must be provided when calling the metric to "
                    "represent the positions of the target class"
                ),
                citation=(
                    "Arias-Duart, Anna, et al. 'Focus! Rating XAI Methods and Finding Biases.'"
                    "arXiv:2109.15035 (2022)"
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
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        For this metric to run we need to get the positions of the target class within the mosaic.
        This should be a np.ndarray containing one tuple per sample, representing the positions
        of the target class within the mosaic (where each tuple contains 0/1 values referring to
        (top_left, top_right, bottom_left, bottom_right).

        An example:
            >> custom_batch=[(1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 1, 0), (0, 1, 0, 1)]

        How to initialise the metric and evaluate explanations by calling the metric instance?
            >> metric = Focus()
            >> scores = {method: metric(**init_params)(model=model,
                           x_batch=x_mosaic_batch,
                           y_batch=y_mosaic_batch,
                           a_batch=None,
                           custom_batch=p_mosaic_batch,
                           **{"explain_func": explain,
                              "explain_func_kwargs": {
                              "method": "GradCAM",
                              "gc_layer": "model._modules.get('conv_2')",
                              "pos_only": True,
                              "interpolate": (2*28, 2*28),
                              "interpolate_mode": "bilinear",}
                              "device": device}) for method in ["GradCAM", "IntegratedGradients"]}

            # Plot example!
            >> metric.plot(results=scores)

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
        custom_batch: any
            Any object that can be passed to the evaluation process.
            Gives flexibility to the user to adapt for implementing their own metric.
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
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        c: np.ndarray = None,
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
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
        c: any
            The custom input to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """

        # Prepare shapes for mosaics.
        self.mosaic_shape = a.shape

        total_positive_relevance = np.sum(a[a > 0], dtype=np.float64)
        target_positive_relevance = 0

        quadrant_functions_list = [
            self.quadrant_top_left,
            self.quadrant_top_right,
            self.quadrant_bottom_left,
            self.quadrant_bottom_right,
        ]

        for quadrant_p, quadrant_func in zip(c, quadrant_functions_list):
            if not bool(quadrant_p):
                continue
            quadrant_relevance = quadrant_func(a=a)
            target_positive_relevance += np.sum(
                quadrant_relevance[quadrant_relevance > 0]
            )

        focus_score = target_positive_relevance / total_positive_relevance

        return focus_score

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Dict[str, Any]:
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
        dictionary[str, np.ndarray]
            Output dictionary with two items:
            1) 'c_batch' as key and custom_batch as value.
            2) 'custom_batch' as key and None as value.
            This results in the keyword argument 'c' being passed to `evaluate_instance()`.

        """
        # Asserts.
        try:
            assert model is not None
            assert x_batch is not None
            assert y_batch is not None
        except AssertionError:
            raise ValueError(
                "Focus requires either a_batch (explanation maps) or "
                "the necessary arguments to compute it for you (model, x_batch & y_batch)."
            )

        # Overwrite custom_batch to have only 'c' as instance input.
        return {"c_batch": custom_batch, "custom_batch": None}

    def quadrant_top_left(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, : int(self.mosaic_shape[1] / 2), : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_a

    def quadrant_top_right(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, int(self.mosaic_shape[1] / 2) :, : int(self.mosaic_shape[2] / 2)
        ]
        return quandrant_a

    def quadrant_bottom_left(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, : int(self.mosaic_shape[1] / 2), int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_a

    def quadrant_bottom_right(self, a: np.ndarray) -> np.ndarray:
        quandrant_a = a[
            :, int(self.mosaic_shape[1] / 2) :, int(self.mosaic_shape[2] / 2) :
        ]
        return quandrant_a
