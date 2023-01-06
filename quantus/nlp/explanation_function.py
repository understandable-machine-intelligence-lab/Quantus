from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Iterable, Tuple, Callable, Optional, Dict, Sequence
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial

from .text_classifier import TextClassifier
from .utils import exponential_kernel, normalize_scores


def _unpack_token_ids_and_attention_mask(
    tokens: Dict[str, tf.Tensor] | tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor | List[None]]:
    if isinstance(tokens, Dict):
        return tokens["input_ids"], tokens["attention_mask"]
    else:
        batch_size = int(tf.shape(tokens)[0])
        return tokens, [None] * batch_size


def tf_explain_grad_norm(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
) -> List[Tuple[List[str], np.ndarray]]:

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)
    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model.forward_pass(embeddings, attention_mask)
        logits_for_label = tf.gather(logits, axis=-1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    grad_norm = tf.linalg.norm(grads, axis=-1)
    scores = normalize_scores(grad_norm.numpy())
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_input_x_gradient(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
) -> List[Tuple[List[str], np.ndarray]]:

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)

    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model.forward_pass(embeddings, attention_mask)
        logits_for_label = tf.gather(logits, axis=1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    scores = tf.math.reduce_sum(embeddings * grads, axis=-1)
    scores = normalize_scores(scores.numpy())

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


@tf.function
def get_interpolated_inputs(
    baseline: tf.Tensor, target: tf.Tensor, num_steps: int
) -> tf.Tensor:
    baseline = tf.cast(baseline, dtype=tf.float64)
    target = tf.cast(target, dtype=tf.float64)
    delta = target - baseline
    scales = tf.linspace(0, 1, num_steps + 1)[:, tf.newaxis, tf.newaxis]
    shape = (num_steps + 1,) + delta.shape
    deltas = scales * tf.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs


def _tf_explain_int_grad(
    *,
    input_ids: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    y: int,
    model: TextClassifier,
    num_steps: int,
    baseline_func: Callable[[tf.Tensor], tf.Tensor],
) -> Tuple[List[str], np.ndarray]:
    embeddings = model.word_embedding_lookup(tf.expand_dims(input_ids, 0))[0]
    baseline = baseline_func(embeddings)

    interpolated_embeddings = get_interpolated_inputs(baseline, embeddings, num_steps)
    interpolated_embeddings = tf.cast(interpolated_embeddings, tf.float32)

    if attention_mask is not None:
        interpolated_attention_mask = tf.stack(
            [attention_mask for _ in range(num_steps + 1)]
        )
    else:
        interpolated_attention_mask = None

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model.forward_pass(
            interpolated_embeddings, interpolated_attention_mask
        )
        logits_for_label = tf.gather(logits, axis=1, indices=y)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    int_grad = tfp.math.trapz(tfp.math.trapz(grads, axis=0))

    scores = normalize_scores(int_grad.numpy())
    return model.tokenizer.convert_ids_to_tokens(input_ids), scores


def tf_explain_int_grad(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *,
    num_steps: int = 10,
    baseline_func: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> List[Tuple[List[str], np.ndarray]]:
    if baseline_func is None:
        baseline_func = lambda x: tf.zeros_like(x)

    tokens = model.tokenizer.tokenize(x_batch)
    batch_size = len(y_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)

    return [
        _tf_explain_int_grad(
            model=model,
            input_ids=input_ids[i],
            attention_mask=attention_mask[i],
            y=y_batch[i],
            num_steps=num_steps,
            baseline_func=baseline_func,
        )
        for i in range(batch_size)
    ]


def sample_masks(
    num_samples: int, num_features: int, seed: Optional[int] = None
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    positions = np.tile(np.arange(num_features), (num_samples, 1))
    permutation_fn = np.vectorize(rng.permutation, signature="(n)->(n)")
    permutations = permutation_fn(positions)  # A shuffled range of positions.
    num_disabled_features = rng.randint(1, num_features + 1, (num_samples, 1))
    # For num_disabled_features[i] == 2, this will set indices 0 and 1 to False.
    return permutations >= num_disabled_features


def mask_input(
    tokens: Sequence[str], masks: np.ndarray, mask_token: str
) -> Iterable[str]:
    """Returns strings with the masked tokens replaced with `mask_token`."""
    for mask in masks:
        parts = [t if mask[i] else mask_token for i, t in enumerate(tokens)]
        yield " ".join(parts)


def _tf_explain_lime(
    *,
    x: str,
    y: int,
    model: TextClassifier,
    mask_token: str,
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    distance_scale: float,
    kernel: Callable[..., np.ndarray],
    seed: int,
    num_samples: int,
    alpha: float,
    solver: str,
) -> Tuple[List[str], np.ndarray]:
    tokens = model.tokenizer.split_into_tokens(x)

    masks = sample_masks(num_samples + 1, len(tokens), seed=seed)
    if not masks.shape[0] == num_samples + 1:
        raise ValueError("Expected num_samples + 1 masks.")

    all_true_mask = np.ones_like(masks[0], dtype=np.bool)
    masks[0] = all_true_mask  # First mask is the full sentence.

    perturbations = list(mask_input(tokens, masks, mask_token))
    outputs = model.predict(perturbations)

    distances = distance_fn(all_true_mask.reshape(1, -1), masks).flatten()
    distances = distance_scale * distances
    distances = kernel(distances)

    # Fit a linear model for the requested output class.
    linear_model = Ridge(alpha=alpha, solver=solver, random_state=seed).fit(
        masks, outputs, sample_weight=distances
    )
    scores = normalize_scores(linear_model.coef_[y])
    return tokens, scores


def tf_explain_lime(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *,
    mask_token: str = "[MASK]",
    distance_scale: float = 100.0,
    kernel: Callable[..., np.ndarray] = exponential_kernel,
    alpha: float = 1.0,
    seed: int = 42,
    solver: str = "cholesky",
    num_samples: int = 3000,
    distance_fn: Callable[..., np.ndarray] = partial(
        pairwise_distances, metric="cosine"
    ),
) -> List[Tuple[List[str], np.ndarray]]:
    return [
        _tf_explain_lime(
            x=x,
            model=model,
            mask_token=mask_token,
            distance_scale=distance_scale,
            kernel=kernel,
            alpha=alpha,
            num_samples=num_samples,
            distance_fn=distance_fn,
            solver=solver,
            seed=seed,
            y=y,
        )
        for x, y in zip(x_batch, y_batch)
    ]


def tf_explain_shap(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *args,
    **kwargs,
) -> List[Tuple[List[str], np.ndarray]]:
    raise NotImplementedError()


def tf_explain_noise_grad(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    n: int = 10,
    m: int = 10,
    explain_fn=tf_explain_grad_norm_over_embeddings,
) -> List[Tuple[List[str], np.ndarray]]:
    """
    - loop [0..m] add gaussian noise to og model parameters
    - loop [0..n] add gaussian noise to input
    - generate [m,n] explanations -> mean over them
    """
    raise NotImplementedError()
