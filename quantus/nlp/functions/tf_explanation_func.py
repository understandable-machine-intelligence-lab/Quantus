"""Explanation functions for TensorFlow models."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import List, Callable, Optional, TYPE_CHECKING, Dict

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    TFHuggingFaceTextClassifier,
)
from quantus.nlp.helpers.types import (
    Explanation,
    NumericalExplainFn,
    ExplainFn,
    NoiseType,
)
from quantus.nlp.helpers.utils import (
    get_interpolated_inputs,
    value_or_default,
    apply_noise,
    apply_to_dict,
    get_embeddings,
    get_input_ids,
)

if TYPE_CHECKING:
    from quantus.nlp.helpers.types import (
        Explanation,
        TF_TensorLike,
    )  # pragma: not covered

    BaselineFn = Callable[[TF_TensorLike], TF_TensorLike]  # pragma: not covered


def tf_explain_gradient_norm(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
) -> List[Explanation]:
    """
    A baseline GradientNorm text-classification explainer.
    The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38.
    GradientNorm explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute L2 norm of gradients.

    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = tf_explain_gradient_norm_numerical(model, embeddings, y_batch, **kwargs)
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_gradient_norm_numerical(
    model: TextClassifier, embeddings: TF_TensorLike, y_batch: np.ndarray, **kwargs
) -> np.ndarray:
    """A version of GradientNorm explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""
    if not isinstance(embeddings, tf.Tensor):
        embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model(embeddings, **kwargs)
        logits_for_label = tf.gather(logits, axis=-1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    return tf.linalg.norm(grads, axis=-1).numpy()


def tf_explain_gradient_x_input(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
) -> List[Explanation]:
    """
    A baseline GradientXInput text-classification explainer.
     The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108.
     GradientXInput explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute vector dot product between input embeddings and gradients.


    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = tf_explain_gradient_x_input_numerical(model, embeddings, y_batch, **kwargs)

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_gradient_x_input_numerical(
    model: TextClassifier, embeddings: TF_TensorLike, y_batch: TF_TensorLike, **kwargs
) -> np.ndarray:
    """A version of GradientXInput explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""
    if not isinstance(embeddings, tf.Tensor):
        embeddings = tf.convert_to_tensor(embeddings, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model(embeddings, **kwargs)
        logits_for_label = tf.gather(logits, axis=1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    return tf.math.reduce_sum(embeddings * grads, axis=-1).numpy()


def tf_explain_integrated_gradients(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
) -> List[Explanation]:
    """
    A baseline Integrated Gradients text-classification explainer. Integrated Gradients explanation algorithm is:
        - Convert inputs to models latent representations.
        - For each x, y in x_batch, y_batch
        - Generate num_steps samples interpolated from baseline to x.
        - Execute forwards pass.
        - Retrieve logits for y.
        - Compute gradient of logits with respect to interpolated samples.
        - Estimate integral over interpolated samples using trapezoid rule.
    In practise, we combine all interpolated samples in one batch, to avoid executing forward and backward passes
    in for-loop. This means potentially, that batch size selected for this XAI method should be smaller than usual.

    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108
    - Sundararajan et al., 2017, Axiomatic Attribution for Deep Networks, https://arxiv.org/pdf/1703.01365.pdf

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    num_steps:
        Number of interpolated samples, which should be generated, default=10.
    baseline_fn:
        Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
        embedding for [UNK] token could be used.
    batch_interpolated_inputs:
        Indicates if interpolated inputs should be stacked into 1 bigger batch.
        This speeds up the explanation, however can be very memory intensive.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    >>> def unknown_token_baseline_function(x):
        ... return tf.convert_to_tensor(np.load(...), dtype=tf.float32)

    >>> tf_explain_integrated_gradients(..., ..., ..., baseline_fn=unknown_token_baseline_function) # noqa

    """
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = tf_explain_integrated_gradients_numerical(
        model,
        embeddings,
        y_batch,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
        batch_interpolated_inputs=batch_interpolated_inputs,
        **kwargs,
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_integrated_gradients_numerical(
    model: TextClassifier,
    embeddings: TF_TensorLike,
    y_batch: TF_TensorLike,
    *,
    num_steps: int = 32,
    baseline_fn: Optional[BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> np.ndarray:
    """A version of Integrated Gradients explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""

    baseline_fn = value_or_default(baseline_fn, lambda: lambda x: tf.zeros_like(x))
    interpolated_embeddings = []

    for i, embeddings_i in enumerate(embeddings):
        interpolated_embeddings.append(
            get_interpolated_inputs(baseline_fn(embeddings_i), embeddings_i, num_steps)
        )

    if batch_interpolated_inputs:
        return _tf_explain_integrated_gradients_batched(
            model, interpolated_embeddings, y_batch, **kwargs
        )
    else:
        return _tf_explain_integrated_gradients_iterative(
            model, interpolated_embeddings, y_batch, **kwargs
        )


def _tf_explain_integrated_gradients_batched(
    model: TextClassifier,
    interpolated_embeddings: List[TF_TensorLike],
    y_batch: TF_TensorLike,
    **kwargs,
) -> np.ndarray:
    interpolated_embeddings = tf.convert_to_tensor(interpolated_embeddings)
    num_steps = interpolated_embeddings.shape[1]

    interpolated_embeddings = tf.reshape(
        tf.cast(interpolated_embeddings, dtype=tf.float32),
        [-1, *interpolated_embeddings.shape[2:]],
    )

    def pseduo_interpolate(x):
        x = tf.broadcast_to(x, (num_steps, *x.shape))
        x = tf.reshape(x, (-1, *x.shape[2:]))
        return x

    interpolated_kwargs = apply_to_dict(kwargs, pseduo_interpolate)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model(interpolated_embeddings, **interpolated_kwargs)
        logits_for_label = tf.gather(logits, axis=-1, indices=y_batch)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    grads = tf.reshape(grads, [len(y_batch), num_steps, *grads.shape[1:]])
    return np.trapz(np.trapz(grads, axis=1), axis=-1)


def _tf_explain_integrated_gradients_iterative(
    model: TextClassifier,
    interpolated_embeddings_batch: List[TF_TensorLike],
    y_batch: TF_TensorLike,
    **kwargs,
) -> np.ndarray:
    scores = []

    for i, interpolated_embeddings in enumerate(interpolated_embeddings_batch):
        interpolated_embeddings = tf.convert_to_tensor(interpolated_embeddings)

        interpolated_kwargs = apply_to_dict(
            {k: v[i] for k, v in kwargs.items()},
            lambda x: tf.broadcast_to(x, (interpolated_embeddings.shape[0], *x.shape)),
        )

        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(interpolated_embeddings, **interpolated_kwargs)
            logits_for_label = tf.gather(logits, axis=-1, indices=y_batch[i])

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        scores.append(np.trapz(np.trapz(grads, axis=0), axis=-1))

    return np.asarray(scores)


def tf_explain_noise_grad_plus_plus(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    *,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    n: int = 10,
    m: int = 10,
    explain_fn: NumericalExplainFn | str = "IntGrad",
    noise_type: NoiseType = NoiseType.multiplicative,
    seed: int = 42,
) -> List[Explanation]:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's weights and model's inputs. The implementation is based
    on https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    sg_mean:
        Mean of normal distribution, from which noise applied to input embeddings is sampled, default=0.0.
    sg_std:
        Standard deviation of normal distribution, from which noise applied to input embeddings is sampled, default=0.4.
    n:
        Number of times noise is applied to weights, default=10.
    m:
        Number of times noise is applied to input embeddings, default=10
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad.
        Otherwise, must have `Callable[[TextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    noise_type:
        If NoiseType.multiplicative weights and input embeddings will be multiplied by noise.
        If NoiseType.additive noise will be added to weights and input embeddings.

    seed:
        PRNG seed used for noise generating distributions.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    >>> import functools
    >>> explain_fn = functools.partial(tf_explain_integrated_gradients_numerical, num_steps=22) # noqa
    >>> tf_explain_noise_grad_plus_plus(..., ..., ..., explain_fn=explain_fn) # noqa

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """

    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = tf_explain_noise_grad_plus_plus_numerical(
        model,
        embeddings,
        y_batch,
        mean=mean,
        std=std,
        sg_mean=sg_mean,
        sg_std=sg_std,
        n=n,
        m=m,
        explain_fn=explain_fn,
        noise_type=noise_type,
        seed=seed,
        **kwargs,
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_noise_grad_plus_plus_numerical(
    model: TextClassifier,
    embeddings: TF_TensorLike,
    y_batch: np.ndarray,
    *,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    n: int = 10,
    m: int = 10,
    explain_fn: NumericalExplainFn | str = "IntGrad",
    noise_type: NoiseType = NoiseType.multiplicative,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """A version of NoiseGrad++ explainer meant for usage with latent space perturbation."""

    if isinstance(explain_fn, str):
        if explain_fn == "NoiseGrad++":
            raise ValueError(
                "Can't use NoiseGrad++ as baseline explanation function for NoiseGrad++"
            )
        if explain_fn not in _numerical_method_mapping:
            raise ValueError(
                f"Unsupported explain_fn {explain_fn}, supported are {list(_numerical_method_mapping.keys())}"
            )
        f = _numerical_method_mapping[explain_fn]
        explain_fn = f
    tf.random.set_seed(seed)

    original_weights = model.weights
    batch_size = embeddings.shape[0]
    num_tokens = embeddings.shape[1]

    explanations = np.zeros(shape=(n, m, batch_size, num_tokens))

    for _n in range(n):
        weights_copy = original_weights.copy()
        for index, params in enumerate(weights_copy):
            params_noise = tf.random.normal(params.shape, mean=mean, stddev=std)
            noisy_params = apply_noise(params, params_noise, noise_type)
            weights_copy[index] = noisy_params

        model.weights = weights_copy
        for _m in range(m):
            inputs_noise = tf.random.normal(
                embeddings.shape, mean=sg_mean, stddev=sg_std
            )
            noisy_embeddings = apply_noise(embeddings, inputs_noise, noise_type)
            explanation = explain_fn(model, noisy_embeddings, y_batch, **kwargs)
            explanations[_n][_m] = explanation

    scores = np.mean(explanations, axis=(0, 1))

    model.weights = original_weights

    return scores


_method_mapping: Dict[str, ExplainFn] = {
    "GradNorm": tf_explain_gradient_norm,
    "GradXInput": tf_explain_gradient_x_input,
    "IntGrad": tf_explain_integrated_gradients,
    "NoiseGrad++": tf_explain_noise_grad_plus_plus,
}

_numerical_method_mapping: Dict[str, NumericalExplainFn] = {
    "GradNorm": tf_explain_gradient_norm_numerical,
    "GradXInput": tf_explain_gradient_x_input_numerical,
    "IntGrad": tf_explain_integrated_gradients_numerical,
    "NoiseGrad++": tf_explain_integrated_gradients_numerical,
}


def tf_explain(
    model: TextClassifier,
    x_batch: List[str] | np.ndarray,
    y_batch: np.ndarray,
    *args,
    method: str,
    **kwargs,
) -> List[Explanation] | np.ndarray:
    """Execute plain text or numerical gradient based explanation methods based on type of inputs provided."""
    if isinstance(x_batch[0], str):
        if method not in _method_mapping:
            raise ValueError(
                f"Unsupported explanation method: {method}, supported are: {list(_method_mapping.keys())}"
            )
        explain_fn = _method_mapping[method]
        return explain_fn(model, x_batch, y_batch, **kwargs)  # noqa

    if method not in _numerical_method_mapping:
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(_numerical_method_mapping.keys())}"
        )
    explain_fn = _numerical_method_mapping[method]  # type: ignore
    return explain_fn(model, x_batch, y_batch, *args, **kwargs)
