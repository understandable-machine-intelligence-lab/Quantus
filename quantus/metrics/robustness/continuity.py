"""This module contains the implementation of the Continuity metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import itertools
import numpy as np

from ..base import PerturbationMetric
from ...helpers import utils
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import translation_x_direction
from ...helpers.similarity_func import lipschitz_constant


class Continuity(PerturbationMetric):
    """
    Implementation of the Continuity test by Montavon et al., 2018.

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        nr_steps: int = 28,
        perturb_baseline: str = "black",
        patch_size: int = 7,
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

        similarity_func (callable): Similarity function applied to compare input and perturbed input.
            If None, the default value is used, default=difference.
        norm_numerator (callable): Function for norm calculations on the numerator.
            If None, the default value is used, default=fro_norm
        norm_denominator (callable): Function for norm calculations on the denominator.
            If None, the default value is used, default=fro_norm
        nr_samples (integer): The number of samples iterated, default=200.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=True.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func (callable): Input perturbation function. If None, the default value is used,
            default=translation_x_direction.
        perturb_baseline (string): Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        patch_size (integer): The patch size for masking, default=7.
        perturb_func_kwargs (dict): Keyword arguments to be passed to perturb_func, default={}.
        nr_steps (integer): The number of steps to iterate over, default=28.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        default_plot_func (callable): Callable that plots the metrics result.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = translation_x_direction

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

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
        if similarity_func is None:
            similarity_func = lipschitz_constant
        self.similarity_func = similarity_func
        self.patch_size = patch_size
        self.nr_steps = nr_steps
        self.nr_patches = None
        self.dx = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "how many patches to split the input image to 'nr_patches', "
                    "the number of steps to iterate over 'nr_steps', the value to replace"
                    " the masking with 'perturb_baseline' and in what direction to "
                    "translate the image 'perturb_func'"
                ),
                citation=(
                    "Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. 'Methods for "
                    "interpreting and understanding deep neural networks.' Digital Signal "
                    "Processing 73, 1-15 (2018"
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
        softmax: bool = False,
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
    ) -> Dict:

        results = {k: [] for k in range(self.nr_patches + 1)}

        for step in range(self.nr_steps):

            # Generate explanation based on perturbed input x.
            dx_step = (step + 1) * self.dx
            x_perturbed = self.perturb_func(
                arr=x,
                indices=np.arange(0, x.size),
                indexed_axes=np.arange(0, x.ndim),
                perturb_dx=dx_step,
                **self.perturb_func_kwargs,
            )
            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)

            # Generate explanations on perturbed input.
            a_perturbed = self.explain_func(
                model=model.get_model(),
                inputs=x_input,
                targets=y,
                **self.explain_func_kwargs,
            )
            # Taking the first element, since a_perturbed will be expanded to a batch dimension
            # not expected by the current index management functions.
            a_perturbed = utils.expand_attribution_channel(a_perturbed, x_input)[0]

            if self.normalise:
                a_perturbed = self.normalise_func(
                    a_perturbed, **self.normalise_func_kwargs
                )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            # Store the prediction score as the last element of the sub_self.last_results dictionary.
            y_pred = float(model.predict(x_input)[:, y])

            results[self.nr_patches].append(y_pred)

            # Create patches by splitting input into grid. Take x_input[0] to avoid batch axis,
            # which a_axes is not tuned for
            axis_iterators = [
                range(0, x_input[0].shape[axis], self.patch_size)
                for axis in self.a_axes
            ]

            for ix_patch, top_left_coords in enumerate(
                itertools.product(*axis_iterators)
            ):
                # Create slice for patch.
                patch_slice = utils.create_patch_slice(
                    patch_size=self.patch_size,
                    coords=top_left_coords,
                )

                a_perturbed_patch = a_perturbed[
                    utils.expand_indices(a_perturbed, patch_slice, self.a_axes)
                ]

                # Taking the first element, since a_perturbed will be expanded to a batch dimension
                # not expected by the current index management functions.
                # a_perturbed = utils.expand_attribution_channel(a_perturbed, x_input)[0]

                if self.normalise:
                    a_perturbed_patch = self.normalise_func(
                        a_perturbed_patch.flatten(), **self.normalise_func_kwargs
                    )

                if self.abs:
                    a_perturbed_patch = np.abs(a_perturbed_patch.flatten())

                # Sum attributions for patch.
                patch_sum = float(sum(a_perturbed_patch))
                results[ix_patch].append(patch_sum)

        return results

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:

        custom_preprocess_batch = [None for _ in x_batch]

        # Get number of patches for input shape (ignore batch and channel dim).
        self.nr_patches = utils.get_nr_patches(
            patch_size=self.patch_size,
            shape=x_batch.shape[2:],
            overlap=True,
        )

        self.dx = np.prod(x_batch.shape[2:]) // self.nr_steps

        # Asserts.
        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)
        asserts.assert_patch_size(patch_size=self.patch_size, shape=x_batch.shape[2:])

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )

    @property
    def aggregated_score(self):
        """
        Implements a continuity correlation score (an addition to the original method) to evaluate the
        relationship between change in explanation and change in function output. It can be seen as an
        quantitative interpretation of visually determining how similar f(x) and R(x1) curves are.
        """
        return np.mean(
            [
                self.similarity_func(
                    self.last_results[sample][self.nr_patches],
                    self.last_results[sample][ix_patch],
                )
                for ix_patch in range(self.nr_patches)
                for sample in self.last_results.keys()
            ]
        )
