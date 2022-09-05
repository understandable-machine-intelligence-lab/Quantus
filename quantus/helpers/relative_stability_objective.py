from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np
from jax import config


config.update("jax_debug_nans", True)


@functools.partial(jax.jit, static_argnums=4)
def _relative_input_stability_objective_single_point(x, x_s, a, a_s, eps_min):
    """
    Computes relative input/representation stabilities maximization objective
    as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors
    """

    nominator = (a - a_s) / (a + (a == 0) * eps_min)  # prevent division by 0
    nominator = jnp.linalg.norm(nominator)

    denominator = x - x_s
    denominator /= x + (x == 0) * eps_min

    denominator = jnp.linalg.norm(denominator)
    denominator = jnp.max(jnp.stack([denominator, eps_min]))

    return nominator / denominator


# Here we batch/vectorize first 4 args of _relative_input_stability_objective_single_point over additional axis at 0,
# which represents a batch of datapoints.
_relative_input_stability_objective_batched = jax.vmap(
    _relative_input_stability_objective_single_point, in_axes=(0, 0, 0, 0, None)
)

# Here we batch/vectorize 2. and 4. argument of _relative_input_stability_objective_batched over an additional axis at 0,
# which represents perturbations of inputs/explanations, and allows us to avoid for-loops.
_relative_input_stability_objective_vectorized_over_perturbation_axis = jax.vmap(
    _relative_input_stability_objective_batched, in_axes=(None, 0, None, 0, None)
)


def relative_input_stability_objective(
    x: np.ndarray,
    xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min=1e-6,
    **kwargs,
) -> np.ndarray | jnp.ndarray:
    """
    Params:
       x:    4D tensor of datapoints with shape (batch_size, ...)
       xs:   5D tensor of datapoints with shape (num_perturbations, batch_size, ...)
       e_x:  4D tensor of explanations with shape (batch_size, ...)
       e_xs: 5D tensor of explanations with shape (num_perturbations, batch_size, ...)
    Returns: A 2D tensor with shape (num_perturbations, batch_size)
    """
    return (
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        _relative_input_stability_objective_vectorized_over_perturbation_axis(x, xs, e_x, e_xs, eps_min)
        .block_until_ready()
    )


def relative_representation_stability_objective(
    l_x: np.ndarray,
    l_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min=1e-6,
    **kwargs,
) -> np.ndarray | jnp.ndarray:
    """
    The required computations are the same as for RIS, so we can just euse it
        Params:
           l_x:  1D tensor of internal model representations for x
           l_xs: 2D tensor of internal model representations for x' (xs)
           e_x:  4D tensor of explanations with shape (batch_size, ...)
           e_xs: 5D tensor of explanations with shape (num_perturbations, batch_size, ...)
        Returns: A 2D tensor with shape (num_perturbations, batch_size)
    """
    return (
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        _relative_input_stability_objective_vectorized_over_perturbation_axis(l_x, l_xs, e_x, e_xs, eps_min)
        .block_until_ready()
    )


@functools.partial(jax.jit, static_argnums=4)
def _relative_output_stability_objective_single_point(x, x_s, a, a_s, eps_min):
    """
    Computes relative output stabilities maximization objective
    as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors.
    Is really similar to _relative_input_stability_objective_single_point(...) up to division in denominator.
    As per https://flax.readthedocs.io/en/latest/advanced_topics/philosophy.html
       "Generally, prefer duplicating code over adding options to functions."
    That's the main reason, this calculation exists as a separate function.
    """

    nominator = (a - a_s) / (a + (a == 0) * eps_min)  # prevent division by 0
    nominator = jnp.linalg.norm(nominator)

    denominator = x - x_s

    denominator = jnp.linalg.norm(denominator)
    denominator = jnp.max(jnp.stack([denominator, eps_min]))

    return nominator / denominator


# Follows the same logic as _relative_input_stability_objective_batched
_relative_output_stability_objective_batched = jax.vmap(
    _relative_output_stability_objective_single_point, in_axes=(0, 0, 0, 0, None)
)

# Follows the same logic as _relative_input_stability_objective_vectorized_over_perturbation_axis
_relative_output_stability_objective_vectorized_over_perturbation_axis = jax.vmap(
    _relative_output_stability_objective_batched, in_axes=(None, 0, None, 0, None)
)


def relative_output_stability_objective(
    h_x: np.ndarray,
    h_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min=1e-6,
    **kwargs,
) -> np.ndarray | jnp.ndarray:
    """
    Params:
       h_x:  1D tensor, with output logits for x with shape (num_classes)
       h_xs: 2D tensor, with output logits for x' (xs) (num_perturbations, num_classes)
       e_x:  4D tensor of explanations with shape (batch_size, ...)
       e_xs: 5D tensor of explanations with shape (num_perturbations, batch_size, ...)
    Returns: A 2D tensor with shape (num_perturbations, batch_size)
    """
    return (
        # https://jax.readthedocs.io/en/latest/async_dispatch.html
        _relative_output_stability_objective_vectorized_over_perturbation_axis(h_x, h_xs, e_x, e_xs, eps_min)
        .block_until_ready()
    )
