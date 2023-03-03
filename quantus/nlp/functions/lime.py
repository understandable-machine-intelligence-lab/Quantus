import functools
from typing import Callable, Iterable, Optional, Sequence, List
import numpy as np
from sklearn import linear_model
from sklearn import metrics

from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import get_input_ids, safe_as_array
from quantus.nlp.helpers.model.text_classifier import TextClassifier


def exponential_kernel(distance: float, kernel_width: float = 25) -> np.ndarray:
    return np.sqrt(np.exp(-(distance**2) / kernel_width**2))


def explain_lime(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    *,
    alpha: float = 1.0,
    solver: str = "cholesky",
    seed: int = 42,
    num_samples: int = 1000,
    mask_token: str = "[UNK]",
    distance_fn: Callable[..., np.ndarray] = functools.partial(
        metrics.pairwise.pairwise_distances, metric="cosine"
    ),
    kernel: Callable[..., np.ndarray] = exponential_kernel,
    distance_scale: float = 100.0,
) -> List[Explanation]:
    """
    LIME explains classifiers by returning a feature attribution score
    for each input feature. It works as follows:

    1) Sample perturbation masks. First the number of masked features is sampled
        (uniform, at least 1), and then that number of features are randomly chosen
        to be masked out (without replacement).
    2) Get predictions from the model for those perturbations. Use these as labels.
    3) Fit a linear model to associate the input positions indicated by the binary
        mask with the resulting predicted label.

    The resulting feature importance scores are the linear model coefficients for
    the requested output class or (in case of regression) the output score.

    This is a reimplementation of the original https://github.com/marcotcr/lime
    and is tested for compatibility. This version supports applying LIME to text input.

    Returns
    -------

    """
    a_batch = []
    input_ids, predict_kwargs = get_input_ids(x_batch, model)

    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        ids = safe_as_array(input_ids[i], force=True)
        tokens = model.convert_ids_to_tokens(ids)
        masks = sample_masks(num_samples + 1, len(tokens), seed=seed)
        assert masks.shape[0] == num_samples + 1, "Expected num_samples + 1 masks."
        all_true_mask = np.ones_like(masks[0], dtype=np.bool)
        masks[0] = all_true_mask

        perturbations = list(get_perturbations(tokens, masks, mask_token))
        logits = model.predict(perturbations)
        outputs = logits[:, y]
        distances = distance_fn(all_true_mask.reshape(1, -1), masks).flatten()
        distances = distance_scale * distances
        distances = kernel(distances)

        # Fit a linear model for the requested output class.
        local_surrogate_model = linear_model.Ridge(
            alpha=alpha, solver=solver, random_state=seed
        ).fit(masks, outputs, sample_weight=distances)

        score = local_surrogate_model.coef_  # noqa
        a_batch.append((tokens, score))
    return a_batch


def sample_masks(num_samples: int, num_features: int, seed: Optional[int] = None):
    rng = np.random.RandomState(seed)
    positions = np.tile(np.arange(num_features), (num_samples, 1))
    permutation_fn = np.vectorize(rng.permutation, signature="(n)->(n)")
    permutations = permutation_fn(positions)  # A shuffled range of positions.
    num_disabled_features = rng.randint(1, num_features + 1, (num_samples, 1))
    return permutations >= num_disabled_features


def get_perturbations(
    tokens: Sequence[str], masks: np.ndarray, mask_token: str
) -> Iterable[str]:
    """Returns strings with the masked tokens replaced with `mask_token`."""
    for mask in masks:
        parts = [t if mask[i] else mask_token for i, t in enumerate(tokens)]
        yield " ".join(parts)
