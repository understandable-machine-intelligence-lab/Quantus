# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Framework-agnostic explanation functions."""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from transformers import pipeline
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.model.tf_hf_model import TFHuggingFaceTextClassifier
from quantus.helpers.model.torch_hf_model import TorchHuggingFaceTextClassifier

from quantus.functions.nlp_explanation_func.lime import explain_lime
from quantus.helpers.types import Explanation, Explanations
from quantus.helpers.utils import (
    add_default_items,
    safe_as_array,
    value_or_default,
)
from quantus.helpers.utils_nlp import map_explanations, is_torch_model, is_tf_model


def explain_shap(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    *,
    batch_size: int = 64,
    init_kwargs: Optional[Dict] = None,
    call_kwargs: Optional[Dict] = None,
) -> List[Explanation]:
    """
    Generate explanations using shapley values. This method depends on shap pip package.

    References
    ----------
        - Lundberg et al., 2017, A Unified Approach to Interpreting Model Predictions, http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
        - https://github.com/slundberg/shap
    """

    if not isinstance(
        model, (TorchHuggingFaceTextClassifier, TFHuggingFaceTextClassifier)
    ):
        raise ValueError(
            "SHAP explanations are only supported for models from HuggingFace Hub"
        )

    import shap

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    call_kwargs = add_default_items(call_kwargs, {"silent": True})

    predict_fn = pipeline(
        "text-classification",
        model=model.get_model(),
        tokenizer=model.tokenizer.unwrap(),
        top_k=None,
        device=getattr(model, "device", None),
    )
    explainer = shap.Explainer(predict_fn, **init_kwargs)

    shapley_values = explainer(x_batch, batch_size=batch_size, **call_kwargs)
    return [(i.feature_names, i.values[:, y]) for i, y in zip(shapley_values, y_batch)]


def generate_text_classification_explanations(
    model: TextClassifier,
    *args,
    method: Optional[str] = None,
    **kwargs,
) -> Explanations:
    """A main 'entrypoint' for calling all text-classification explanation functions available in Quantus."""

    if method is None:
        warnings.warn(
            f"Using quantus 'explain' function as an explainer without specifying 'method' (string) "
            f"in kwargs will produce a simple 'GradientNorm' explanation.\n",
            category=UserWarning,
        )
        method = "GradNorm"

    if "device" in kwargs:
        kwargs.pop("device")

    if method == "LIME":
        return explain_lime(model, *args, **kwargs)
    if method == "SHAP":
        return explain_shap(model, *args, **kwargs)

    if is_tf_model(model):
        from quantus.functions.nlp_explanation_func.tf_explanation_func import (
            tf_explain,
        )

        return tf_explain(model, *args, method=method, **kwargs)

    if is_torch_model(model):
        from quantus.functions.nlp_explanation_func.torch_explanation_func import (
            torch_explain,
        )

        result = torch_explain(model, *args, method=method, **kwargs)
        return map_explanations(result, safe_as_array)  # noqa

    raise ValueError(f"Unable to identify DNN framework of the model.")
