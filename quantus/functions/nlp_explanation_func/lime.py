# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

import functools
from typing import Callable, List, Optional, Sequence, NamedTuple

import numpy as np
from sklearn import linear_model, metrics

from quantus.helpers.collection_utils import value_or_default
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.types import Explanation

__all__ = ["explain_lime", "LimeConfig"]


class LimeConfig(NamedTuple):
    alpha: float = 1.0
    solver: str = "cholesky"
    seed: int = 42
    num_samples: int = 1000
    mask_token: str = "[UNK]"
    distance_fn: Callable = functools.partial(
        metrics.pairwise.pairwise_distances, metric="cosine"
    )
    kernel: Optional[Callable] = None
    distance_scale: float = 100.0


def explain_lime(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    config: Optional[LimeConfig] = None,
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
    config = value_or_default(config, lambda: LimeConfig())
    kernel = value_or_default(config.kernel, lambda: exponential_kernel)
    a_batch = []
    input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)

    for i, (x, y) in enumerate(zip(x_batch, y_batch)):
        ids = input_ids[i]
        tokens = model.tokenizer.convert_ids_to_tokens(ids)
        masks = sample_masks(config.num_samples + 1, len(tokens), seed=config.seed)
        assert (
            masks.shape[0] == config.num_samples + 1
        ), "Expected num_samples + 1 masks."
        all_true_mask = np.ones_like(masks[0], dtype=bool)
        masks[0] = all_true_mask

        perturbations = get_perturbations(tokens, masks, config.mask_token)
        logits = model.predict(perturbations)
        outputs = logits[:, y]
        # fmt: off
        distances = config.distance_fn(all_true_mask.reshape(1, -1), masks).flatten()  # noqa
        # fmt: on
        distances = config.distance_scale * distances
        distances = kernel(distances)

        # Fit a linear model for the requested output class.
        local_surrogate_model = linear_model.Ridge(
            alpha=config.alpha, solver=config.solver, random_state=config.seed
        ).fit(masks, outputs, sample_weight=distances)

        score = local_surrogate_model.coef_  # noqa
        a_batch.append((tokens, score))
    return a_batch


# ---------------------- internal -----------------------
def sample_masks(num_samples: int, num_features: int, seed: Optional[int] = None):
    rng = np.random.RandomState(seed)
    positions = np.tile(np.arange(num_features), (num_samples, 1))
    permutation_fn = np.vectorize(rng.permutation, signature="(n)->(n)", cache=True)
    permutations = permutation_fn(positions)  # A shuffled range of positions.
    num_disabled_features = rng.randint(1, num_features + 1, (num_samples, 1))
    return permutations >= num_disabled_features


def get_perturbations(
    tokens: Sequence[str], masks: np.ndarray, mask_token: str
) -> List[str]:
    """Returns strings with the masked tokens replaced with `mask_token`."""
    result = []
    for mask in masks:
        parts = [t if mask[i] else mask_token for i, t in enumerate(tokens)]
        result.append(" ".join(parts))
    return result


def exponential_kernel(distance: float, kernel_width: float = 25) -> np.ndarray:
    return np.sqrt(np.exp(-(distance**2) / kernel_width**2))
