"""This module contains the implementation of the Selectivity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import itertools
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantus.functions.perturb_func import baseline_replacement_by_indices, batch_baseline_replacement_by_indices
from quantus.helpers import plotting, utils, warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class Selectivity(Metric[List[float]]):
    """
    Implementation of Selectivity test by Montavon et al., 2018.

    At each iteration, a patch (e.g., of size 4 x 4) corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    Assumptions:
        - The original metric definition relies on perturbation functionality suited only for images.
        Therefore, only apply the metric to 3-dimensional (image) data. To extend the applicablity
        to other data domains, adjustments to the current implementation might be necessary.

    References:
        1) Grégoire Montavon et al.:
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Selectivity"
    data_applicability = {DataType.IMAGE}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        patch_size: int = 8,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        patch_size: integer
            The patch size for masking, default=8.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
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

        if default_plot_func is None:
            default_plot_func = plotting.plot_selectivity_experiment

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

        if perturb_func is None:
            perturb_func = batch_baseline_replacement_by_indices

        # Save metric-specific attributes.
        self.patch_size = patch_size
        self.perturb_func = make_perturb_func(perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline)

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline' and the patch size for masking" " 'patch_size'"),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73 (2018): 1-15"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

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
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean([utils.calculate_auc(np.array(curve)) for i, curve in enumerate(self.evaluation_scores)])

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> List[List[float]]:
        """
        This method performs XAI evaluation on a single batch of explanations.
        For more information on the specific logic, we refer the metric’s initialisation docstring.

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        kwargs:
            Unused.

        Returns
        -------
        scores_batch:
            The evaluation results.
        """
        # Prepare shapes. Expand a_batch if not the same shape
        if x_batch.shape != a_batch.shape:
            a_batch = np.broadcast_to(a_batch, x_batch.shape)
        x_batch_shape = x_batch.shape
        if len(x_batch.shape) == 3:
            x_batch = x_batch[:, :, None]
            a_batch = a_batch[:, :, None]

        batch_size = a_batch.shape[0]

        x_perturbed = x_batch.copy()

        # Pad input and attributions. This is needed to allow for any patch_size.
        x_perturbed_h, x_perturbed_w = x_perturbed.shape[-2:]
        padding_h, padding_w = utils.get_padding_size(x_perturbed_h, self.patch_size), utils.get_padding_size(
            x_perturbed_w, self.patch_size
        )
        padding = ((0, 0), (0, 0), padding_h, padding_w)
        x_pad = utils._pad_array(
            x_batch,
            padding,
            mode="constant",
            padded_axes=np.arange(len(x_perturbed.shape)),
        )
        a_pad = utils._pad_array(
            a_batch,
            padding,
            mode="constant",
            padded_axes=np.arange(len(x_perturbed.shape)),
        )

        # Get patch indices of sorted attributions (descending).
        att_sums_list = []
        patches_list = []
        for block_indices in utils.get_block_indices(x_pad, self.patch_size):
            # Create slice for patch.
            a_sum = a_pad.reshape(batch_size, -1)[np.arange(batch_size)[:, None], block_indices].sum(axis=-1)

            # Sum attributions for patch.
            att_sums_list.append(a_sum)
            patches_list.append(block_indices)
        att_sums = np.stack(att_sums_list, -1)
        patches = np.stack(patches_list, 1)

        # Create ordered list of patches.
        order = np.argsort(-att_sums, -1)
        ordered_patches = patches[np.arange(batch_size)[:, None], order].transpose(1, 0, 2)

        # Increasingly perturb the input and store the decrease in function value.
        results = []
        x_perturbed_pad = utils._pad_array(
            x_perturbed,
            padding,
            mode="edge",
            padded_axes=np.arange(len(x_perturbed.shape)),
        )
        x_perturbed_pad_shape = x_perturbed_pad.shape
        for patch_slice in ordered_patches:
            # Perturb.
            x_perturbed_pad = self.perturb_func(arr=x_perturbed_pad.reshape(batch_size, -1), indices=patch_slice)

            # Remove padding.
            x_perturbed_pad = x_perturbed_pad.reshape(*x_perturbed_pad_shape)
            x_perturbed = x_perturbed_pad[
                :,
                :,
                padding_h[0] : x_perturbed_pad.shape[2] - padding_h[1],
                padding_w[0] : x_perturbed_pad.shape[3] - padding_w[1],
            ]

            # Check if the perturbation caused change
            for x_element, x_perturbed_element in zip(x_batch, x_perturbed):
                warn.warn_perturbation_caused_no_change(x=x_element, x_perturbed=x_perturbed_element)

            # Predict on perturbed input x.
            x_input = model.shape_input(x_perturbed, x_batch_shape, channel_first=True, batched=True)
            y_pred_perturb = model.predict(x_input)[np.arange(batch_size), y_batch]

            results.append(y_pred_perturb)
        results = np.stack(results, 1).astype(np.float64)

        return results
