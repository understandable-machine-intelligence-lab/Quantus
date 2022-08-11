from typing import Union

import numpy as np
import jax.numpy as jnp
import jax
import functools
from abc import abstractmethod

from quantus import Metric, utils, asserts, perturb_func, ModelInterface
from typing import Callable, Sequence, Tuple

from tqdm import tqdm


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, None))
def relative_stability(
    x: jnp.ndarray, x_s: jnp.ndarray, a: jnp.ndarray, a_s: np.ndarray, eps_min
) -> float:
    """
    Parameters
            x: an input image with
            x_s: a perturbed input image
            a: attribution for x: explanation, logits, or model internal state
            a_s: attribution for x_s: explanation, logits, or model internal state
            eps_min: prevents division by zero

        Returns
            float:
    """
    # Prevents division by 0
    a += eps_min
    x += eps_min

    nominator = (a - a_s) / a
    nominator = jnp.linalg.norm(nominator)

    denominator = (x - x_s) / x
    denominator = jnp.linalg.norm(denominator)
    denominator = jnp.max(denominator, initial=eps_min)

    return nominator / denominator


relative_stability_vectorized_over_attributions = jax.vmap(
    relative_stability, in_axes=(None, 0, None, 0, None)
)


class RelativeStability(Metric):

    DEFAULT_EPS_MIN = 1e-6
    DEFAULT_NUM_PERTURBATIONS = 100
    name: str
    objective_to_maximize = relative_stability_vectorized_over_attributions


    def __init__(self, *args, **kwargs):
        """
        Implementation of RIS according to https://arxiv.org/pdf/2203.06877.pdf
        Parameters:
            kwargs:
               eps_min (optional): a small constant to prevent denominator from being 0, default 1e-6
               num_perturbations(optional): number of times perturb_func should be executed, default 10
        """

        super().__init__(*args, **kwargs)

        self.eps_min = kwargs.pop("eps_min", self.DEFAULT_EPS_MIN)

        self.num_perturbations = kwargs.pop("num_perturbations", self.DEFAULT_NUM_PERTURBATIONS)

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        *args,
        **kwargs
    ) -> Union[float, Sequence]:

        """
        A base class for relative stability metrics from https://arxiv.org/pdf/2203.06877.pdf


        Parameters:
            model:
            x_batch: batch data points
            y_batch: batch of labels for x_batch
            perturb_func(optional): a function used to perturbate inputs, must be provided unless no xs_batch provided
            xs_batch(optional): a batch of perturbed inputs

            exp_fun (optional): a function to generate explanations, must be provided unless a_batch, as_batch were not provided
            a_batch(optional): pre-computed explanations for x
            as_batch(optional): pre-computed explanations for x'

            kwargs: kwargs, which are passed to perturb_func, explain_func and quantus.

        For each image x:
         - generate N perturbed x' in the neighborhood of x (or use pre-computed)
         - find x' which results in the same label
         - Compute (or use pre-computed) explanations e_x and e_x'
         - Compute relative stability objective, find max value with regard to x'
         - In practise we just use max over a finite batch of x'
        """


        # Reshape input batch to channel first order, if needed
        x_batch, channel_first = utils.move_channel_axis_batch(x_batch, **kwargs)
        self.model = utils.get_wrapped_model(model, channel_first)

        xs_batch = self.get_perturbed_inputs(x_batch, y_batch, model, **kwargs)

        arg, arg_s = self._get_stability_argument(x_batch, xs_batch, **kwargs)
        ex, exs = self.get_explanations(model, x_batch, y_batch, xs_batch)

        result = relative_stability_vectorized_over_attributions(arg, arg_s, ex, exs, self.eps_min)

        result = jnp.max(result, axis=0).to_py()
        if self.return_aggregate:
            result = self.aggregate_func(result)

        self.all_results.append(result)
        self.last_results = [result]

        return result




    def get_perturbed_inputs(self, x_batch, y_batch, model, **kwargs) -> np.ndarray:
        if (
            "perturb_func" in kwargs and "xs_batch" in kwargs
        ) or (
            "perturb_func" not in kwargs and "xs_batch" not in kwargs
        ):
            raise ValueError(
                "Must provide either perturb_func or xs_batch in kwargs"
            )


        if 'perturb_func' in kwargs:

            perturb_function: Callable = kwargs.pop(
                "perturb_func", perturb_func.random_noise
            )
            asserts.assert_perturb_func(perturb_function)

            xs_batch = []
            it = range(self.num_perturb)

            if self.display_progressbar:
                it = tqdm(it, desc=f"Collecting perturbation for {self.name}")

            for _ in it:
                xs = perturb_function(x_batch, **kwargs)
                logits = model.predict(xs)
                labels = np.argmax(logits, axis=1)

                same_label_indexes = np.argwhere(labels == y_batch)
                xs = xs[same_label_indexes].reshape(-1, *xs.shape[1:])
                xs_batch.append(xs)

            # pull all new images into 0 axes
            xs_batch = np.asarray(xs_batch).reshape(-1, *x_batch.shape[1:])
            # drop images, which cause dims not to be divisible
            xs_batch = xs_batch[: xs_batch.shape[0] // x_batch.shape[0] * x_batch.shape[0]]
            # make xs_batch have the same shape as x_batch, with new batching axis at 0
            xs_batch = xs_batch.reshape(-1, *x_batch.shape)
            return xs_batch

        xs_batch = kwargs.pop('xs_batch')
        if len(xs_batch.shape) <= len(x_batch.shape):
            raise ValueError('xs_batch must have 1 more batch axis than x_batch')

        return xs_batch


    def get_explanations(self, model: ModelInterface, x_batch, y_batch, xs_batch, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if ('explain_func' in kwargs and ('a_batch' in kwargs or 'as_batch' in kwargs)
        ) or ('explain_func' not in kwargs and ('a_batch' not in kwargs or 'as_batch' in kwargs)):
            raise ValueError('Must provide either explain_func or (a_batch and as_batch)')

        if 'explain_func' in kwargs:
            explain_func: Callable = kwargs.pop('explain_func')

            a_batch = explain_func(
                model=model.get_model(), inputs=x_batch, targets=y_batch, **kwargs
            )

            it = xs_batch
            if self.display_progressbar:
                it = tqdm(it, desc=f'Collecting explanations for {self.name}')

            as_batch = [
                explain_func(model=model.get_model(), inputs=i, targets=y_batch, **kwargs) for i in it
            ]

        else:
            a_batch = kwargs.pop('a_batch')
            as_batch = kwargs.pop('as_batch')

        if self.normalise:
            a_batch = self.normalise_func(a_batch)
            as_batch = [self.normalise_func(i) for i in as_batch]

        if self.abs:
            a_batch = self.abs(a_batch)
            as_batch = [self.abs(i) for i in as_batch]


        return a_batch, np.asarray(as_batch)


    @abstractmethod
    def _get_stability_argument(self, x, xs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        The only non-generic part among all 3 Relative Stability metrics
        """
        pass




class RelativeInputStability(RelativeStability):

    name = 'Relative Input Stability'

    """
    RIS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{x - x'}{x}||_p, \epsilon_{min})}

    The numerator of the metric measures the `p norm of the percent change of explanation ex' on the perturbed
        instance x'  with respect to the explanation ex on the original point x,
        the denominator measures the `p norm between (normalized) inputs x and x'
        and the max term prevents division by zero in cases when norm || (x−x')/x ||_p is less than
        some small epsilon_min>0
    """

    def _get_stability_argument(self, x, xs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Here it's just the x and x'
        """
        return x, xs



class RelativeOutputStability(RelativeStability):

    name = 'Relative Output Stability'

    """
    ROS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{h(x) - h(x')}||_p, \epsilon_{min})}
    where h(x) and h(x') are the output logits for x and x', respectively

    """

    def _get_stability_argument(self, x, xs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        hx = self.model.predict(x)
        hxs = self.model.predict(xs)
        return hx, hxs


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, None))
def relative_representation_stability(
    l: jnp.ndarray, l_s: jnp.ndarray, a: jnp.ndarray, a_s: np.ndarray, eps_min
) -> float:
    """
    Parameters
            x: an input image with
            x_s: a perturbed input image
            a: attribution for x: explanation, logits, or model internal state
            a_s: attribution for x_s: explanation, logits, or model internal state
            eps_min: prevents division by zero

        Returns
            float:
    """
    # Prevents division by 0
    a += eps_min

    nominator = (a - a_s) / a
    nominator = jnp.linalg.norm(nominator)

    denominator = l - l_s
    denominator = jnp.linalg.norm(denominator)
    denominator = jnp.max(denominator, initial=eps_min)

    return nominator / denominator


relative_representation_stability_vectorized_over_attributions = jax.vmap(
    relative_representation_stability, in_axes=(None, 0, None, 0, None)
)


class RelativeRepresentationStability(RelativeStability):

    name = 'Relative Representation Stability'

    """
    RRS(x, x', ex, ex') = max \frac{||\frac{e_x - e_{x'}}{e_x}||_p}{max (||\frac{L_x - L_{x'}}{L_x}||_p, \epsilon_{min})}

    where L(·) denotes the internal model representation, e.g., output embeddings of hidden layers.

    """
    objective_to_maximize = relative_representation_stability_vectorized_over_attributions


    def _get_stability_argument(self, x, xs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterate over layers (except the last one), collect the output of each layer
        Stack them in a tensor, or any other work around ???
        """
        lx = self.model.get_hidden_layers_outputs(x)
        lx = np.stack(lx)

        lxs = [self.model.get_hidden_layers_outputs(i) for i in xs]
        lxs = [np.stack(i) for i in lxs]

        return lx, np.asarray(lxs)












