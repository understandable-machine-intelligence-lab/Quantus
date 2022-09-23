"""This module contains the implementation of the Selectivity metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import itertools
import numpy as np

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers import utils
from ...helpers import plotting
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import baseline_replacement_by_indices


class Selectivity(PerturbationMetric):
    """
    Implementation of Selectivity test by Montavon et al., 2018.

    At each iteration, a patch (e.g., of size 4 x 4) corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
    a large number of examples.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.
    """

    @asserts.attributes_check
    def __init__(
        self,
        patch_size: int = 8,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        patch_size (integer): The patch size for masking, default=8.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
            default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_selectivity_experiment

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        self.patch_size = patch_size

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
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

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        custom_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict[str, Any]] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        softmax: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
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
        i: int,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        c: Any,
        p: Any,
    ) -> List[float]:

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        patches = []
        x_perturbed = x.copy()

        # Pad input and attributions. This is needed to allow for any patch_size.
        pad_width = self.patch_size - 1
        x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=self.a_axes)
        a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=self.a_axes)

        # Get patch indices of sorted attributions (descending).
        att_sums = []
        axis_iterators = [
            range(pad_width, x_pad.shape[axis] - pad_width) for axis in self.a_axes
        ]
        for top_left_coords in itertools.product(*axis_iterators):

            # Create slice for patch.
            patch_slice = utils.create_patch_slice(
                patch_size=self.patch_size,
                coords=top_left_coords,
            )

            # Sum attributions for patch.
            att_sums.append(
                a_pad[utils.expand_indices(a_pad, patch_slice, self.a_axes)].sum()
            )
            patches.append(patch_slice)

            # Create ordered list of patches.
        ordered_patches = [patches[p] for p in np.argsort(att_sums)[::-1]]

        # Remove overlapping patches.
        blocked_mask = np.zeros(x_pad.shape, dtype=bool)
        ordered_patches_no_overlap = []
        for patch_slice in ordered_patches:
            patch_mask = np.zeros(x_pad.shape, dtype=bool)
            patch_mask_exp = utils.expand_indices(patch_mask, patch_slice, self.a_axes)
            patch_mask[patch_mask_exp] = True
            intersected = blocked_mask & patch_mask

            if not intersected.any():
                ordered_patches_no_overlap.append(patch_slice)
                blocked_mask = blocked_mask | patch_mask

        # Increasingly perturb the input and store the decrease in function value.
        results = np.array([None for _ in range(len(ordered_patches_no_overlap))])
        for patch_id, patch_slice in enumerate(ordered_patches_no_overlap):

            # Pad x_perturbed. The mode should depend on the used perturb_func.
            x_perturbed_pad = utils._pad_array(
                x_perturbed, pad_width, mode="edge", padded_axes=self.a_axes
            )

            # Perturb the input.
            x_perturbed_pad = self.perturb_func(
                arr=x_perturbed_pad,
                indices=patch_slice,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )

            # Remove padding.
            x_perturbed = utils._unpad_array(
                x_perturbed_pad, pad_width, padded_axes=self.a_axes
            )

            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x and store the difference from predicting on unperturbed input.
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])

            results[patch_id] = y_pred_perturb

        return results

    @property
    def get_auc_score(self):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean(
            [
                utils.calculate_auc(np.array(curve))
                for i, curve in enumerate(self.last_results)
            ]
        )
