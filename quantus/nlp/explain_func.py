from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Iterable, Tuple, Callable, Optional, Dict

from .text_classifier import TextClassifier, Tokenizer


def _unpack_token_ids_and_attention_mask(
    tokens: Dict[str, tf.Tensor] | tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor | List[None]]:
    if isinstance(tokens, Dict):
        return tokens["input_ids"], tokens["attention_mask"]
    else:
        batch_size = int(tf.shape(tokens)[0])
        return tokens, [None] * batch_size


@tf.function(reduce_retracing=True)
def tf_normalise_attributions(x: tf.Tensor) -> tf.Tensor:
    abs_x = tf.abs(x)
    max_x = tf.reduce_max(abs_x)
    return abs_x / max_x


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
    model: TextClassifier,
    input_ids: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    target: int,
    tokenizer: Tokenizer,
    num_steps: int,
    baseline_func: Callable[[tf.Tensor], tf.Tensor],
) -> Tuple[List[str], np.ndarray]:
    embeddings = model.embedding_lookup(tf.expand_dims(input_ids, 0))[0]
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
        logits_for_label = tf.gather(logits, axis=1, indices=target)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    int_grad = tfp.math.trapz(tfp.math.trapz(grads, axis=0))

    scores = tf_normalise_attributions(int_grad).numpy()
    return tokenizer.convert_ids_to_tokens(input_ids.numpy()), scores


def tf_explain_int_grad(
    model: TextClassifier,
    text: List[str],
    targets: np.ndarray | Iterable[float],
    tokenizer: Tokenizer,
    num_steps: int = 10,
    baseline_func: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> List[Tuple[List[str], np.ndarray]]:

    if baseline_func is None:
        baseline_func = lambda x: tf.zeros_like(x)

    tokens = tokenizer.tokenize(text)
    batch_size = len(targets)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)

    return [
        _tf_explain_int_grad(
            model,
            input_ids[i],
            attention_mask[i],
            targets[i],
            tokenizer,
            num_steps,
            baseline_func,
        )
        for i in range(batch_size)
    ]


def _tf_explain_grad_norm(
    model: TextClassifier,
    input_ids: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    target: int,
    tokenizer: Tokenizer,
) -> Tuple[List[str], np.ndarray]:
    embeddings = model.embedding_lookup(input_ids)
    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model.forward_pass(embeddings, attention_mask)
        logits_for_label = tf.gather(logits, axis=1, indices=target)

    grads = tape.gradient(logits_for_label, embeddings)
    grad_norm = tf.linalg.norm(grads, axis=-1)
    scores = tf_normalise_attributions(grad_norm[0]).numpy()
    return tokenizer.convert_ids_to_tokens(input_ids), scores


def tf_explain_grad_norm(
    model: TextClassifier,
    text: List[str],
    targets: np.ndarray | Iterable[float],
    tokenizer: Tokenizer,
) -> List[Tuple[List[str], np.ndarray]]:
    tokens = tokenizer.tokenize(text)
    batch_size = len(targets)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)

    return [
        _tf_explain_grad_norm(
            model, input_ids[i], attention_mask[i], targets[i], tokenizer
        )
        for i in range(batch_size)
    ]


def tf_explain_lime(
    model: TextClassifier,
    text: List[str],
    targets: np.ndarray | Iterable[float],
    tokenizer: Tokenizer,
) -> List[Tuple[List[str], np.ndarray]]:
    pass
