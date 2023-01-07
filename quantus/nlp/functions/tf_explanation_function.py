from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Iterable, Tuple, Callable, Optional, Dict, Sequence
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from transformers import pipeline
from shap import Explainer


from quantus.nlp.helpers.utils import exponential_kernel, normalize_scores
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    HuggingFaceTextClassifierTF,
)


def _unpack_token_ids_and_attention_mask(
    tokens: Dict[str, np.ndarray] | tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor | List[None]]:
    if isinstance(tokens, Dict):
        return tokens["input_ids"], tokens["attention_mask"]
    else:
        batch_size = int(tf.shape(tokens)[0])
        return tokens, [None] * batch_size


def tf_explain_gradient_norm(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
) -> List[Explanation]:

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)
    scores = tf_explain_gradient_norm_over_embeddings(
        embeddings, attention_mask, y_batch, model
    )
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_gradient_norm_over_embeddings(
    embeddings: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
) -> np.ndarray:

    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model.forward(embeddings, attention_mask)
        logits_for_label = tf.gather(logits, axis=-1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    grad_norm = tf.linalg.norm(grads, axis=-1)
    scores = normalize_scores(grad_norm.numpy())
    return scores


def tf_explain_input_x_gradient(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
) -> List[Explanation]:

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)

    scores = tf_explain_input_x_gradient_over_embeddings(
        embeddings, attention_mask, y_batch, model
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_input_x_gradient_over_embeddings(
    embeddings: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    y_batch: np.ndarray | tf.Tensor,
    model: TextClassifier,
) -> np.ndarray:
    with tf.GradientTape() as tape:
        tape.watch(embeddings)
        logits = model.forward(embeddings, attention_mask)
        logits_for_label = tf.gather(logits, axis=1, indices=y_batch)

    grads = tape.gradient(logits_for_label, embeddings)
    scores = tf.math.reduce_sum(embeddings * grads, axis=-1)
    scores = normalize_scores(scores.numpy())
    return scores


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


def _tf_explain_integrated_gradients_over_embeddings(
    *,
    embedding: tf.Tensor,
    attention_mask: Optional[tf.Tensor],
    y: int,
    model: TextClassifier,
    num_steps: int,
    baseline_fn: Callable[[tf.Tensor], tf.Tensor],
) -> np.ndarray:

    baseline = baseline_fn(embedding)
    interpolated_embeddings = get_interpolated_inputs(baseline, embedding, num_steps)
    interpolated_embeddings = tf.cast(interpolated_embeddings, tf.float32)
    interpolated_attention_mask = (
        tf.stack([attention_mask for _ in range(num_steps + 1)])
        if attention_mask is not None
        else None
    )

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model.forward(interpolated_embeddings, interpolated_attention_mask)
        logits_for_label = tf.gather(logits, axis=1, indices=y)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    int_grad = tfp.math.trapz(tfp.math.trapz(grads, axis=0))

    scores = normalize_scores(int_grad.numpy())
    return scores


def tf_explain_integrated_gradients(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> List[Explanation]:
    if baseline_fn is None:
        baseline_fn = lambda x: tf.zeros_like(x)

    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)

    scores = tf_explain_integrated_gradients_over_embeddings(
        embeddings,
        attention_mask,
        y_batch,
        model,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
    )

    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
    ]


def tf_explain_integrated_gradients_over_embeddings(
    embeddings: tf.Tensor,
    attention_mask: tf.Tensor,
    y_batch: np.ndarray | Iterable[float],
    model: TextClassifier,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> np.ndarray:

    if baseline_fn is None:
        baseline_fn = lambda x: tf.zeros_like(x)
    batch_size = len(y_batch)

    return np.asarray(
        [
            _tf_explain_integrated_gradients_over_embeddings(
                model=model,
                embedding=embeddings[i],
                attention_mask=attention_mask[i],
                y=y_batch[i],
                num_steps=num_steps,
                baseline_fn=baseline_fn,
            )
            for i in range(batch_size)
        ]
    )


def tf_explain_noise_grad_plus_plus(
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
    explain_fn: Callable[
        [tf.Tensor, Optional[tf.Tensor], np.ndarray | tf.Tensor, TextClassifier],
        np.ndarray,
    ] = tf_explain_gradient_norm_over_embeddings,
    noise_type: str = "multiplicative",
    seed: int = 42,
) -> List[Explanation]:

    if noise_type not in ("multiplicative", "additive"):
        raise ValueError(f"Unsupported noise type: {noise_type}")

    tf.random.set_seed(seed)

    weights = model.get_weights()
    tokens = model.tokenizer.tokenize(x_batch)
    input_ids, attention_mask = _unpack_token_ids_and_attention_mask(tokens)
    embeddings = model.word_embedding_lookup(input_ids)
    batch_size = embeddings.shape[0]
    num_tokens = embeddings.shape[1]
    explanations = np.zeros(shape=(n, m, batch_size, num_tokens))

    for _n in range(n):
        weights_copy = weights.copy()
        for index, params in enumerate(weights_copy):
            params_noise = tf.random.normal(params.shape, mean=mean, stddev=std)
            noisy_params = (
                params * params_noise
                if noise_type == "multiplicative"
                else params + params_noise
            )
            weights_copy[index] = noisy_params

        model.set_weights(weights_copy)
        for _m in range(m):
            inputs_noise = tf.random.normal(
                embeddings.shape, mean=sg_mean, stddev=sg_std
            )
            noisy_embeddings = (
                embeddings * inputs_noise
                if noise_type == "multiplicative"
                else embeddings + inputs_noise
            )
            explanation = explain_fn(noisy_embeddings, attention_mask, y_batch, model)
            explanations[_n][_m] = explanation

    scores = np.mean(explanations, axis=(0, 1))

    model.set_weights(weights)
    return [
        (model.tokenizer.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)
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
) -> Explanation:
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
    scores = normalize_scores(linear_model.coef_[y])  # noqa
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
) -> List[Explanation]:
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
    *,
    display_progressbar=False,
    init_kwargs: Optional[Dict] = None,
    call_kwargs: Optional[Dict] = None,
) -> List[Explanation]:
    if init_kwargs is None:
        init_kwargs = {}
    if call_kwargs is None:
        call_kwargs = {}
    if isinstance(model, HuggingFaceTextClassifierTF):
        return _tf_explain_shap_huggingface(
            x_batch,
            y_batch,
            model,
            display_progressbar=display_progressbar,
            init_kwargs=init_kwargs,
            call_kwargs=call_kwargs,
        )
    raise NotImplementedError()


def _tf_explain_shap_huggingface(
    x_batch: List[str],
    y_batch: np.ndarray | Iterable[float],
    model: HuggingFaceTextClassifierTF,
    *,
    display_progressbar,
    init_kwargs: Dict,
    call_kwargs: Dict,
) -> List[Explanation]:
    predict_fn = pipeline(
        "text-classification",
        model=model.model,
        tokenizer=model.tokenizer.tokenizer,  # noqa
        top_k=None,
    )
    shapley_values = Explainer(predict_fn, **init_kwargs)(
        x_batch, silent=not display_progressbar, **call_kwargs
    )
    return [(i.feature_names, i.values[:, y]) for i, y in zip(shapley_values, y_batch)]


_method_mapping = {
    "GradNorm": tf_explain_gradient_norm,
    "InputXGrad": tf_explain_input_x_gradient,
    "IntGrad": tf_explain_integrated_gradients,
    "LIME": tf_explain_lime,
    "SHAP": tf_explain_shap,
    "NoiseGrad++": tf_explain_noise_grad_plus_plus,
}


def tf_explain(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
    method: str,
    **kwargs,
) -> List[Explanation]:
    if method is None:
        raise ValueError(f"Please provide explanation method name in `name` kwarg")
    explain_fn = _method_mapping.get(method)
    if explain_fn is None:
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(_method_mapping.keys())}"
        )
    return explain_fn(x_batch, y_batch, model, **kwargs)
