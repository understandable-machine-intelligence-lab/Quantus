"""This module contains the implementation of the Symmetric Relevance Gain metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from quantus.functions.perturb_func import batch_baseline_replacement_by_indices
from quantus.helpers import asserts, utils, warn
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
class SymmetricRelevanceGain(Metric[List[float]]):
    """
    Implementation of the Symmetric Relevance Gain (SRG) by Blücher et al., 2024.

    SRG runs two pixel-flipping experiments (Bach et al., 2015; Samek et al., 2017) that
    share one feature ordering: most influential first (MIF, descending attribution) and
    its exact reverse, least influential first (LIF). The per-sample score is the area
    between the two prediction curves,

        SRG = AUC(LIF curve) − AUC(MIF curve),

    which equals the sum of the two relevance gains MRG and LRG; the AUC of the random
    ordering baseline cancels in the difference and never has to be estimated. SRG
    rankings are largely insensitive to the occlusion strategy (baseline value, step
    size), which resolves the disagreement problem between the MIF and LIF benchmarks.

    Higher is better; a random attribution scores 0 in expectation, and with
    softmax outputs (default) scores lie in [−1, 1].

    Deviations from the paper, following Quantus conventions:
        - Features are flattened input entries grouped by the sorted attribution order
          (`n_steps`/`features_in_step`), not superpixels. Attributions are broadcast
          over the channel axis, so each pixel of a (C, H, W) image appears as C tied
          features; with `features_in_step >= C` this closely matches flipping whole
          pixels.
        - The tracked class is the user-supplied `y_batch`, not the model's prediction
          on the unoccluded input. For an exact paper replication pass
          `y_batch=model(x).argmax(1)`.

    References:
        1) Stefan Blücher et al.: "Decoupling Pixel Flipping and Occlusion Strategy for
        Consistent XAI Benchmarks." Transactions on Machine Learning Research (2024).
        https://openreview.net/forum?id=bIiLXdtUVM
        2) Wojciech Samek et al.: "Evaluating the visualization of what a deep neural
        network has learned." IEEE Transactions on Neural Networks and Learning
        Systems 28.11 (2017): 2660-2673.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Symmetric Relevance Gain"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        n_steps: int = 28,
        features_in_step: Optional[int] = None,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: Union[float, int, str, np.ndarray] = "mean",
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
        n_steps: integer
            The number of occlusion steps per curve; the group size is derived as
            ceil(n_features / n_steps) and the last group may be smaller, default=28.
            Ignored if features_in_step is given.
        features_in_step: integer, optional
            The exact number of flattened features occluded per step (must divide the
            number of features). Overrides n_steps, default=None.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution,
            default=False. Note that SRG is designed for signed attributions;
            abs=True changes the semantics of the metric.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=batch_baseline_replacement_by_indices. With the default, the
            baseline is computed once per batch from the unperturbed input; a custom
            function is called per step on the partially perturbed input and disables
            the torch fast path.
        perturb_baseline: float, int, str, np.ndarray
            Indicates the type of baseline: "mean", "uniform", "black" or "white",
            default="mean".
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
        self.n_steps = n_steps
        self.features_in_step = features_in_step
        self.perturb_baseline = perturb_baseline
        self.use_default_perturb_func = (
            perturb_func is None and perturb_func_kwargs is None
        )
        if perturb_func is None:
            perturb_func = batch_baseline_replacement_by_indices
        self.perturb_func = make_perturb_func(
            perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline
        )

        # Resolved per call in custom_preprocess (depends on the input shape).
        self.features_in_step_: int = features_in_step or 0

        # Per-call curve storage, populated by evaluate_batch (see last_mif_curves/last_lif_curves).
        self._mif_curves_batches: List[np.ndarray] = []
        self._lif_curves_batches: List[np.ndarray] = []

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline' and the step granularity "
                    "'n_steps'/'features_in_step' (SRG rankings are designed to be "
                    "robust to both); also note that 'abs=True' discards the signed "
                    "attribution information SRG evaluates symmetrically"
                ),
                citation=(
                    "Blücher, Stefan, Vielhaben, Johanna, and Strodthoff, Nils. 'Decoupling Pixel "
                    "Flipping and Occlusion Strategy for Consistent XAI Benchmarks.' Transactions "
                    "on Machine Learning Research (2024)"
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
        output labels (y_batch) and a torch model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module
            A torch model that is subject to explanation.
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
            >> metric = SymmetricRelevanceGain(normalise=False)
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

    def custom_preprocess(
        self,
        x_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Resolves the per-step group size from n_steps (or validates an explicitly
        passed features_in_step against the flattened feature count) and resets the
        per-call curve storage.

        Parameters
        ----------
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        kwargs:
            Unused.

        Returns
        -------
        None
        """
        n_features = int(np.prod(x_batch.shape[1:]))
        if self.features_in_step is not None:
            asserts.assert_features_in_step(
                features_in_step=self.features_in_step,
                input_shape=x_batch.shape[1:],
            )
            self.features_in_step_ = self.features_in_step
        else:
            self.features_in_step_ = math.ceil(n_features / self.n_steps)

        self._mif_curves_batches = []
        self._lif_curves_batches = []

    @property
    def last_mif_curves(self) -> Optional[np.ndarray]:
        """MIF prediction curves of the last call, shape (n_samples, n_steps + 1) incl. the unoccluded point."""
        if not self._mif_curves_batches:
            return None
        return np.concatenate(self._mif_curves_batches, axis=0)

    @property
    def last_lif_curves(self) -> Optional[np.ndarray]:
        """LIF prediction curves of the last call, shape (n_samples, n_steps + 1) incl. the unoccluded point."""
        if not self._lif_curves_batches:
            return None
        return np.concatenate(self._lif_curves_batches, axis=0)

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ) -> List[float]:
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
        # Prepare shapes. Expand a_batch if not the same shape.
        if x_batch.shape != a_batch.shape:
            a_batch = np.broadcast_to(a_batch, x_batch.shape)

        batch_size = a_batch.shape[0]
        a_flat = a_batch.reshape(batch_size, -1)

        # One descending sort; the LIF ordering is its exact reverse so that ties are
        # broken consistently between the two curves.
        order_mif = np.argsort(-a_flat, axis=1, kind="stable")

        if self._can_use_torch_fast_path(model):
            curves_mif, curves_lif = self._compute_curves_torch(
                model, x_batch, y_batch, order_mif
            )
        else:
            curves_mif, curves_lif = self._compute_curves_numpy(
                model, x_batch, y_batch, order_mif
            )

        self._mif_curves_batches.append(curves_mif)
        self._lif_curves_batches.append(curves_lif)

        # The shared endpoints (unoccluded and fully occluded) cancel in the AUC
        # difference, so SRG reduces to the mean over the after-step differences.
        srg = (curves_lif[:, 1:] - curves_mif[:, 1:]).mean(axis=1)
        return srg.tolist()

    def _step_slices(self, n_features: int) -> List[slice]:
        """Contiguous chunks of the sorted feature order, one per occlusion step."""
        fis = self.features_in_step_
        n_steps = math.ceil(n_features / fis)
        return [
            slice(step * fis, min((step + 1) * fis, n_features))
            for step in range(n_steps)
        ]

    def _compute_curves_numpy(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        order_mif: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the MIF and LIF prediction curves with numpy-side perturbation,
        shape (batch_size, n_steps + 1) each, including the shared unoccluded point.
        """
        batch_size = x_batch.shape[0]
        single_shape = x_batch.shape[1:]
        n_features = int(np.prod(single_shape))
        order_lif = order_mif[:, ::-1]

        x_mif = x_batch.reshape(batch_size, -1).astype(float)
        x_lif = x_mif.copy()

        baseline = None
        if self.use_default_perturb_func:
            # Compute the baseline once from the unperturbed input and reuse it for all
            # steps and both curves (the paper's constant imputer).
            baseline = utils.get_baseline_value(
                value=self.perturb_baseline,
                arr=x_mif,
                return_shape=x_mif.shape,
                batched=True,
            )

        # Shared unoccluded curve point.
        x_input = model.shape_input(
            x_batch, x_batch.shape, channel_first=True, batched=True
        )
        p_0 = model.predict(x_input)[np.arange(batch_size), y_batch]
        preds_mif, preds_lif = [p_0], [p_0]

        for sl in self._step_slices(n_features):
            ix_mif, ix_lif = order_mif[:, sl], order_lif[:, sl]
            if baseline is not None:
                np.put_along_axis(
                    x_mif, ix_mif, np.take_along_axis(baseline, ix_mif, axis=1), axis=1
                )
                np.put_along_axis(
                    x_lif, ix_lif, np.take_along_axis(baseline, ix_lif, axis=1), axis=1
                )
            else:
                x_mif = self.perturb_func(arr=x_mif, indices=ix_mif)
                x_lif = self.perturb_func(arr=x_lif, indices=ix_lif)

            # One forward pass per step for both curves.
            x_cat = np.concatenate([x_mif, x_lif]).reshape(
                2 * batch_size, *single_shape
            )
            x_input = model.shape_input(
                x_cat, x_cat.shape, channel_first=True, batched=True
            )
            preds = model.predict(x_input)[
                np.arange(2 * batch_size), np.tile(y_batch, 2)
            ]
            preds_mif.append(preds[:batch_size])
            preds_lif.append(preds[batch_size:])

        return np.stack(preds_mif, axis=1), np.stack(preds_lif, axis=1)

    def _can_use_torch_fast_path(self, model: ModelInterface) -> bool:
        """The torch-resident fast path applies to plain torch modules with the default perturbation."""
        if not self.use_default_perturb_func:
            return False
        try:
            from quantus.helpers.model.pytorch_model import (
                PyTorchModel,
                safe_isinstance,
            )
        except ImportError:
            return False
        return isinstance(model, PyTorchModel) and not safe_isinstance(
            model.get_model(), "transformers.modeling_utils.PreTrainedModel"
        )

    def _compute_curves_torch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        order_mif: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Torch-resident equivalent of _compute_curves_numpy: the running perturbed
        inputs, baseline and orderings stay on-device, with one H2D copy up front and
        one D2H copy at the end.
        """
        import torch

        if model.get_model().training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")

        batch_size = x_batch.shape[0]
        single_shape = x_batch.shape[1:]
        n_features = int(np.prod(single_shape))

        x_flat = x_batch.reshape(batch_size, -1)
        baseline = utils.get_baseline_value(
            value=self.perturb_baseline,
            arr=x_flat,
            return_shape=x_flat.shape,
            batched=True,
        )

        device = model.device
        forward = model.get_softmax_arg_model()
        predict_kwargs = model.model_predict_kwargs

        with torch.no_grad():
            x = torch.as_tensor(x_flat, dtype=torch.float32, device=device)
            base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
            idx_mif = torch.as_tensor(
                np.ascontiguousarray(order_mif, dtype=np.int64), device=device
            )
            idx_lif = torch.as_tensor(
                np.ascontiguousarray(order_mif[:, ::-1], dtype=np.int64), device=device
            )
            y = torch.as_tensor(np.asarray(y_batch), dtype=torch.int64, device=device)
            y_cat = y.repeat(2)
            rows = torch.arange(2 * batch_size, device=device)

            # Shared unoccluded curve point.
            p_0 = forward(x.reshape(batch_size, *single_shape), **predict_kwargs)[
                torch.arange(batch_size, device=device), y
            ]
            preds_mif, preds_lif = [p_0], [p_0]

            x_mif, x_lif = x.clone(), x.clone()
            for sl in self._step_slices(n_features):
                ix_mif, ix_lif = idx_mif[:, sl], idx_lif[:, sl]
                x_mif.scatter_(1, ix_mif, base.gather(1, ix_mif))
                x_lif.scatter_(1, ix_lif, base.gather(1, ix_lif))

                # One forward pass per step for both curves.
                x_cat = torch.cat([x_mif, x_lif]).reshape(2 * batch_size, *single_shape)
                preds = forward(x_cat, **predict_kwargs)[rows, y_cat]
                preds_mif.append(preds[:batch_size])
                preds_lif.append(preds[batch_size:])

            curves_mif = torch.stack(preds_mif, dim=1).cpu().numpy()
            curves_lif = torch.stack(preds_lif, dim=1).cpu().numpy()

        return curves_mif, curves_lif
