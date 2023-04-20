# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Framework-agnostic explanation functions."""
from __future__ import annotations

import logging
from importlib import util
from typing import NamedTuple, List

import numpy as np
from transformers import pipeline

from quantus.functions.nlp.lime import explain_lime
from quantus.helpers.collection_utils import safe_as_array, value_or_default
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.nlp_utils import map_explanations, is_transformers_available
from quantus.helpers.tf_utils import is_tensorflow_model, is_tensorflow_available
from quantus.helpers.torch_utils import is_torch_available, is_torch_model
from quantus.helpers.types import Explanation

try:
    from quantus.helpers.model.tensor_rt_model import TensorRTModel
except Exception:
    TensorRTModel = type(None)

log = logging.getLogger(__name__)

if is_tensorflow_available():
    from transformers_gradients.text_classification import huggingface, tensor_rt

    tf_explain_mapping = {
        "huggingface": {
            "GradNorm": huggingface.gradient_norm,
            "GradXInput": huggingface.gradient_x_input,
            "IntGrad": huggingface.integrated_gradients,
            "NoiseGrad": huggingface.noise_grad,
            "NoiseGrad++": huggingface.noise_grad_plus_plus,
        },
        "tensor_rt": {
            "GradNorm": tensor_rt.gradient_norm,
            "GradXInput": tensor_rt.gradient_x_input,
            "IntGrad": tensor_rt.integrated_gradients,
            "NoiseGrad": tensor_rt.noise_grad,
            "NoiseGrad++": tensor_rt.noise_grad_plus_plus,
        },
    }

if is_torch_available():
    from quantus.functions.nlp.torch_explanation_func import (
        torch_explain,
    )

if is_transformers_available():
    from transformers.utils.hub import PushToHubMixin


def is_shap_available() -> bool:
    return util.find_spec("shap") is not None


if is_shap_available():
    import shap

__all__ = ["ShapConfig", "generate_text_classification_explanations"]


class ShapConfig(NamedTuple):
    max_evals: int = 500
    seed: int = 42
    batch_size: str | int = 64
    silent: bool = True


def explain_shap(
    model: TextClassifier,
    x_batch: list[str],
    y_batch: np.ndarray,
    *,
    config: ShapConfig | None = None,
) -> list[Explanation]:
    """
    Generate explanations using shapley values. This method depends on shap pip package.

    References
    ----------
        - Lundberg et al., 2017, A Unified Approach to Interpreting Model Predictions,
            https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
        - https://github.com/slundberg/shap
    """

    if not is_shap_available():
        raise ValueError("SHAP requires `shap` package installation.")

    if not (
        is_transformers_available() or isinstance(model.get_model(), PushToHubMixin)
    ):
        raise ValueError(
            "SHAP explanations are only supported for models from HuggingFace Hub"
        )

    config = value_or_default(config, lambda: ShapConfig())

    predict_fn = pipeline(
        "text-classification",
        model=model.get_model(),
        tokenizer=model.tokenizer.tokenizer,
        top_k=None,
        device=getattr(model, "device", None),
    )
    explainer = shap.Explainer(predict_fn, seed=config.seed)

    shapley_values = explainer(
        x_batch,
        batch_size=config.batch_size,
        max_evals=config.max_evals,
        silent=config.silent,
    )
    return [(i.feature_names, i.values[:, y]) for i, y in zip(shapley_values, y_batch)]


def generate_text_classification_explanations(
    model: TextClassifier,
    x_batch: List[str] | np.ndarray,
    y_batch: np.ndarray,
    method: str | None = None,
    **kwargs,
) -> list[Explanation] | np.ndarray:
    """A main 'entrypoint' for calling all text-classification explanation functions available in Quantus."""

    if method is None:
        log.warning(
            f"Using quantus 'explain' function as an explainer without specifying 'method' (string) "
            f"in kwargs will produce a simple 'GradientNorm' explanation.\n",
        )
        method = "GradNorm"

    if "device" in kwargs:
        # device is saved in model instance.
        kwargs.pop("device")

    if method == "LIME":
        return explain_lime(model, x_batch, y_batch, **kwargs)
    if method == "SHAP":
        return explain_shap(model, x_batch, y_batch, **kwargs)

    if is_tensorflow_model(model) or isinstance(model, TensorRTModel):
        if "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        else:
            attention_mask = None

        if isinstance(x_batch[0], str):
            tokenizer = model.tokenizer.tokenizer
        else:
            tokenizer = None

        if isinstance(model, TensorRTModel):
            if method not in tf_explain_mapping["tensor_rt"]:
                raise ValueError(
                    f"Unsupported explanation function, supported are {list(tf_explain_mapping['tensor_rt'].keys())}"
                )

            def embed_func(bla, ids):
                return model.embedding_lookup(ids)

            model_func = model.embeddings_model
            fn = tf_explain_mapping["tensor_rt"][method]

            return fn(
                model_func,
                x_batch,
                y_batch,
                attention_mask,
                tokenizer=tokenizer,
                embeddings_lookup_fn=embed_func,
                **kwargs,
            )

        if method not in tf_explain_mapping["huggingface"]:
            raise ValueError(
                f"Unsupported explanation function, supported are {list(tf_explain_mapping['huggingface'].keys())}"
            )

        fn = tf_explain_mapping["huggingface"][method]

        return fn(
            model.get_model(),
            x_batch,
            y_batch,
            attention_mask,
            tokenizer=tokenizer,
            **kwargs,
        )

    if is_torch_model(model):
        result = torch_explain(model, x_batch, y_batch, method=method, **kwargs)
        return map_explanations(result, safe_as_array)  # noqa

    raise ValueError(f"Unable to identify DNN framework of the model.")
