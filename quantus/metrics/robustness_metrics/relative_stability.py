import warnings
from typing import Union, Dict, Callable, Tuple
import numpy as np
import jax.numpy as jnp
import jax
from abc import abstractmethod, ABC
import functools
from tqdm import tqdm

from quantus.metrics.base import Metric
from quantus.helpers import utils, perturb_func


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def relative_stability_objective(
    x: jnp.ndarray,
    x_s: jnp.ndarray,
    a: jnp.ndarray,
    a_s: jnp.ndarray,
    eps_min: float,
    division_in_denominator: bool,
    denominator_norm_axis: Union[Tuple[int, int], int],
) -> jnp.ndarray:
    """
    Parameters
            x: an input image / logits for x / internal representation for x
            x_s: a perturbed input image / logits for x_s / internal representation for x_s
            a: attribution for x: explanation, logits, or model internal state
            a_s: attribution for x_s: explanation, logits, or model internal state
            eps_min: prevents division by zero

        Returns
            float:
    """

    # Prevent division by 0
    x += eps_min
    a += eps_min

    nominator = (a - a_s) / a
    nominator = jnp.linalg.norm(nominator, axis=(1, 2))
    nominator = nominator.reshape(-1)

    denominator = x - x_s
    if division_in_denominator:
        denominator /= x

    denominator = jnp.linalg.norm(denominator, axis=denominator_norm_axis)
    denominator = denominator.reshape(-1)

    eps_arr = jnp.full(denominator.shape, eps_min)

    denominator = jnp.stack([denominator, eps_arr])
    denominator = jnp.max(denominator, axis=0)

    return nominator / denominator


relative_stability_objective_vectorized_over_perturbation_axis = jax.vmap(
    relative_stability_objective, in_axes=(None, 0, None, 0, None, None, None)
)


class RelativeStability(Metric, ABC):
    DEFAULT_EPS_MIN = 1e-6
    DEFAULT_NUM_PERTURBATIONS = 100
    name: str

    def __init__(self, *args, **kwargs):
        """
        A base class for relative stability metrics from https://arxiv.org/pdf/2203.06877.pdf
        Parameters:
            kwargs:
               eps_min (optional): a small constant to prevent denominator from being 0, default 1e-6
               num_perturbations(optional): number of times perturb_func should be executed, default 10
        """

        super().__init__(*args, **kwargs)

        self.eps_min = kwargs.pop("eps_min", self.DEFAULT_EPS_MIN)
        self.num_perturbations = kwargs.pop(
            "num_perturbations", self.DEFAULT_NUM_PERTURBATIONS
        )

    def __call__(
        self, model, x_batch: np.ndarray, y_batch: np.ndarray, *args, **kwargs
    ) -> Union[float, np.ndarray]:

        """
        Parameters:
            model:
            x_batch: batch data points
            y_batch: batch of labels for x_batch

            perturb_func(optional): a function used to perturbate inputs, must be provided unless no xs_batch provided
            xs_batch(optional): a batch of perturbed inputs

            exp_fun (optional): a function to generate explanations, must be provided unless a_batch, as_batch were not provided
            a_batch(optional): pre-computed explanations for x
            as_batch(optional): pre-computed explanations for x'

            kwargs: kwargs, which are passed to perturb_func, explain_func and quantus.Metric base class.

        For each image x:
         - generate N perturbed x' in the neighborhood of x
            - find x' which results in the same label
            - or use pre-computed
         - Compute (or use pre-computed) explanations e_x and e_x'
         - Compute relative stability objective, find max value with regard to x'
         - In practise we just use max over a finite batch of x'
        """

        channel_first = utils.infer_channel_first(x_batch)
        self.model = utils.get_wrapped_model(model, channel_first)

        if "a_batch" in kwargs and "as_batch" in kwargs and "xs_batch" not in kwargs:
            raise ValueError(
                "When providing pre-computed explanations, must also provide x' (xs_batch)"
            )

        xs_batch, kwargs = self._get_perturbed_inputs(x_batch, y_batch, **kwargs)
        ex, exs, kwargs = self._get_explanations(x_batch, y_batch, xs_batch, **kwargs)

        result = self._compute_objective(x_batch, xs_batch, ex, exs)

        result = jnp.max(result, axis=0)

        if self.return_aggregate:
            result = self.aggregate_func(result)

        self.all_results.append(result)
        self.last_results = [result]

        return result

    def _get_perturbed_inputs(
        self, x_batch, y_batch, **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            xs_batch: of perturbed inputs
            kwargs: Dict of updated kwargs
        """

        if "xs_batch" not in kwargs:

            if "perturb_func" not in kwargs:
                warnings.warn(
                    'No "perturb_func" provided, using random noise as default'
                )

            perturb_function: Callable = kwargs.pop(
                "perturb_func", perturb_func.random_noise
            )

            xs_batch = []
            it = range(self.num_perturbations)

            if self.display_progressbar:
                it = tqdm(it, desc=f"Collecting perturbation for {self.name}")

            for _ in it:
                xs = perturb_function(x_batch, **kwargs)
                logits = self.model.predict(xs)
                labels = np.argmax(logits, axis=1)

                same_label_indexes = np.argwhere(labels == y_batch)
                xs = xs[same_label_indexes].reshape(-1, *xs.shape[1:])
                xs_batch.append(xs)

            # pull all new images into 0 axes
            xs_batch = np.vstack(xs_batch)
            # drop images, which cause dims not to be divisible
            xs_batch = xs_batch[
                : xs_batch.shape[0] // x_batch.shape[0] * x_batch.shape[0]
            ]
            # make xs_batch have the same shape as x_batch, with new batching axis at 0
            xs_batch = xs_batch.reshape(-1, *x_batch.shape)
            return xs_batch, kwargs

        xs_batch = kwargs.pop("xs_batch")
        if len(xs_batch.shape) <= len(x_batch.shape):
            raise ValueError("xs_batch must have 1 more batch axis than x_batch")

        return xs_batch, kwargs

    def _get_explanations(
        self, x_batch, y_batch, xs_batch, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Returns:
            a_batch: batch of explanations
            as_batch: batch of batches of perturbed explanations
            kwargs: Dict of updated kwargs
        """
        if (
            "explain_func" in kwargs and ("a_batch" in kwargs or "as_batch" in kwargs)
        ) or (
            "explain_func" not in kwargs
            and ("a_batch" not in kwargs or "as_batch" not in kwargs)
        ):
            raise ValueError(
                "Must provide either explain_func or (a_batch and as_batch)"
            )

        if "explain_func" in kwargs:
            explain_func: Callable = kwargs.pop("explain_func")

            a_batch = explain_func(
                model=self.model.get_model(), inputs=x_batch, targets=y_batch, **kwargs
            )

            it = xs_batch
            if self.display_progressbar:
                it = tqdm(it, desc=f"Collecting explanations for {self.name}")

            as_batch = [
                explain_func(
                    model=self.model.get_model(), inputs=i, targets=y_batch, **kwargs
                )
                for i in it
            ]

        else:
            a_batch = kwargs.pop("a_batch")
            as_batch = kwargs.pop("as_batch")

            if len(as_batch.shape) <= len(a_batch.shape):
                raise ValueError(
                    "Batch of perturbed explanations must have 1 more axis"
                )

        if self.normalise:
            a_batch = self.normalise_func(a_batch)
            as_batch = [self.normalise_func(i) for i in as_batch]

        if self.abs:
            a_batch = np.abs(a_batch)
            as_batch = np.abs(as_batch)

        return a_batch, np.asarray(as_batch), kwargs

    @abstractmethod
    def _compute_objective(self, x, xs, e, es) -> np.ndarray:
        """
        The only non-generic part among all 3 Relative Stability metrics
        """
        pass


class RelativeInputStability(RelativeStability):
    name = "Relative Input Stability"

    """
    RIS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{x - x'}{x}||_p, \epsilon_{min})}

    The numerator of the metric measures the `p norm of the percent change of explanation ex' on the perturbed
        instance x'  with respect to the explanation ex on the original point x,
        the denominator measures the `p norm between (normalized) inputs x and x'
        and the max term prevents division by zero in cases when norm || (x−x')/x ||_p is less than
        some small epsilon_min>0
    """

    def _compute_objective(self, x, xs, e, es) -> np.ndarray:
        result = relative_stability_objective_vectorized_over_perturbation_axis(
            jnp.asarray(x),
            jnp.asarray(xs),
            jnp.asarray(e),
            jnp.asarray(es),
            self.eps_min,
            True,
            (1, 2),
        )
        return result.to_py()  # noqa


class RelativeOutputStability(RelativeStability):
    name = "Relative Output Stability"

    """
    ROS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{h(x) - h(x')}||_p, \epsilon_{min})}
       where h(x) and h(x') are the output logits for x and x', respectively
    """

    def _compute_objective(self, x, xs, e, es) -> np.ndarray:
        hx = self.model.predict(x)
        hxs = [self.model.predict(i) for i in xs]
        result = relative_stability_objective_vectorized_over_perturbation_axis(
            jnp.asarray(hx),
            jnp.asarray(hxs),
            jnp.asarray(e),
            jnp.asarray(es),
            self.eps_min,
            False,
            1,
        )
        return result.to_py()  # noqa


class RelativeRepresentationStability(RelativeStability):
    name = "Relative Representation Stability"

    """
    RRS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{L_x - L_{x'}}{L_x}||_p, \epsilon_{min})}
       where L(·) denotes the internal model representation, e.g., output embeddings of hidden layers.
    """

    def _compute_objective(self, x, xs, e, es) -> np.ndarray:
        lx = self.model.get_hidden_layers_outputs(x)
        lxs = [self.model.get_hidden_layers_outputs(i) for i in xs]
        lxs = jnp.asarray(lxs)
        result = relative_stability_objective_vectorized_over_perturbation_axis(
            jnp.asarray(lx),
            jnp.asarray(lxs),
            jnp.asarray(e),
            jnp.asarray(es),
            self.eps_min,
            True,
            1,
        )
        return result.to_py()  # noqa
