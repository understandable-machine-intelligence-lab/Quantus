"""This module contains the implementation of the Random Logit metric."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import Metric
from ...helpers import asserts
from ...helpers import warn_func
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.similarity_func import ssim


class RandomLogit(Metric):
    """
    Implementation of the Random Logit Metric by Sixt et al., 2020.

    The Random Logit Metric computes the distance between the original explanation and a reference explanation of
    a randomly chosen non-target class.

    References:
        1) Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. "When Explanations Lie: Why Many Modified BP
        Attributions Fail."arXiv preprint, arXiv:1912.09818v6 (2020)
    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Callable = None,
        num_classes: int = 1000,
        seed: int = 42,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
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
        similarity_func (callable): Similarity function applied to compare input and perturbed input,
            default=ssim.
        num_classes (integer): Number of prediction classes in the input, default=1000.
        seed (int): Seed used for the random generator, default=42.
        abs (boolean): Indicates whether absolute operation is applied on the attribution, default=False.
        normalise (boolean): Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func (callable): Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs (dict): Keyword arguments to be passed to normalise_func on call, default={}.
        return_aggregate (boolean): Indicates if an aggregated score should be computed over all instances.
        aggregate_func (callable): Callable that aggregates the scores given an evaluation call.
        default_plot_func (callable): Callable that plots the metrics result.
        disable_warnings (boolean): Indicates whether the warnings are printed, default=False.
        display_progressbar (boolean): Indicates whether a tqdm-progress-bar is printed, default=False.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

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
            similarity_func = ssim
        self.similarity_func = similarity_func
        self.num_classes = num_classes
        self.seed = seed

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("similarity metric 'similarity_func'"),
                citation=(
                    "Sixt, Leon, Granz, Maximilian, and Landgraf, Tim. 'When Explanations Lie: "
                    "Why Many Modified BP Attributions Fail.' arXiv preprint, "
                    "arXiv:1912.09818v6 (2020)"
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
    ) -> float:

        # Randomly select off-class labels.
        np.random.seed(self.seed)
        y_off = np.array(
            [
                np.random.choice(
                    [y_ for y_ in list(np.arange(0, self.num_classes)) if y_ != y]
                )
            ]
        )

        # Explain against a random class.
        a_perturbed = self.explain_func(
            model=model.get_model(),
            inputs=np.expand_dims(x, axis=0),
            targets=y_off,
            **self.explain_func_kwargs,
        )

        # Normalise and take absolute values of the attributions, if True.
        if self.normalise:
            a_perturbed = self.normalise_func(a_perturbed, **self.normalise_func_kwargs)

        if self.abs:
            a_perturbed = np.abs(a_perturbed)

        return self.similarity_func(a.flatten(), a_perturbed.flatten())

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

        # Additional explain_func assert, as the one in general_preprocess()
        # won't be executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
