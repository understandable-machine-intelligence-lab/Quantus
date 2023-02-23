"""Framework-agnostic explanation functions."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
import warnings
from transformers import pipeline
from functools import partial

from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import (
    value_or_default,
    safe_isinstance,
)

TF_HuggingfaceModelClass = "quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier.TFHuggingFaceTextClassifier"
Torch_HuggingfaceModelClass = "quantus.nlp.helpers.model.torch_huggingface_text_classifier.TorchHuggingFaceTextClassifier"


def explain_lime(
    model: TextClassifier,
    x_batch: List[str],
    y_batch: np.ndarray,
    *,
    batch_size: int = 64,
    init_kwargs: Optional[Dict] = None,
    call_kwargs: Optional[Dict] = None,
) -> List[Explanation]:
    """
    Generate explanations using LIME method. This method depends on lime pip package.

    References
    ----------
        - Marco Túlio Ribeiro et al., 2016, "Why Should I Trust You?": Explaining the Predictions of Any Classifier, https://arxiv.org/pdf/1602.04938.pdf
        - https://github.com/marcotcr/lime
    """

    from lime.lime_text import LimeTextExplainer  # noqa

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    call_kwargs = value_or_default(call_kwargs, lambda: {})
    predict_fn = partial(model.predict, batch_size=batch_size)

    explainer = LimeTextExplainer(mask_string="[MASK]", **init_kwargs)

    explanations = []
    for x, y in zip(x_batch, y_batch):
        ex = explainer.explain_instance(
            x, predict_fn, top_labels=1, **call_kwargs
        ).as_list(label=y)
        explanations.append(([i[0] for i in ex], np.asarray([i[1] for i in ex])))

    return explanations


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

    import shap  # noqa
    import shap.maskers

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    call_kwargs = value_or_default(call_kwargs, lambda: {})
    predict_fn = partial(model.predict, batch_size=batch_size)

    if safe_isinstance(model, [TF_HuggingfaceModelClass, Torch_HuggingfaceModelClass]):
        predict_fn = pipeline(
            "text-classification",
            model=model.model,  # type: ignore
            tokenizer=model.tokenizer.tokenizer,  # type: ignore
            top_k=None,
            device=getattr(model, "device", None),
        )
        explainer = shap.Explainer(predict_fn, **init_kwargs)
    else:
        explainer = shap.PartitionExplainer(
            predict_fn, shap.maskers.Text(), **init_kwargs  # noqa
        )

    shapley_values = explainer(
        x_batch, silent=True, batch_size=batch_size, **call_kwargs
    )
    return [(i.feature_names, i.values[:, y]) for i, y in zip(shapley_values, y_batch)]


def explain(
    model: TextClassifier,
    x_batch: List[str] | np.ndarray,
    y_batch: np.ndarray,
    *args,
    method: Optional[str] = None,
    framework: Optional[str] = None,
    **kwargs,
) -> List[Explanation]:
    """A main 'entrypoint' for calling all text-classification explanation functions available in Quantus."""

    if method is None:
        warnings.warn(
            f"Using quantus 'explain' function as an explainer without specifying 'method' (string) "
            f"in kwargs will produce a simple 'GradientNorm' explanation.\n",
            category=UserWarning,
        )
        method = "GradNorm"

    if method == "LIME":
        return explain_lime(model, x_batch, y_batch, **kwargs)
    if method == "SHAP":
        return explain_shap(model, x_batch, y_batch, **kwargs)

    if safe_isinstance(model, TF_HuggingfaceModelClass):
        from .tf_explanation_func import tf_explain

        return tf_explain(model, x_batch, y_batch, *args, method=method, **kwargs)

    if safe_isinstance(model, Torch_HuggingfaceModelClass):
        from .torch_explanation_func import torch_explain

        return torch_explain(model, x_batch, y_batch, *args, method=method, **kwargs)

    internal_model = None
    for i in ("model", "_model"):
        if hasattr(model, i):
            internal_model = getattr(model, i)
            break

    if safe_isinstance(internal_model, "keras.Model"):
        from .tf_explanation_func import tf_explain

        return tf_explain(model, x_batch, y_batch, *args, method=method, **kwargs)

    if safe_isinstance(internal_model, "torch.nn.Module"):
        from .torch_explanation_func import torch_explain

        return torch_explain(model, x_batch, y_batch, *args, method=method, **kwargs)

    if framework is None:
        raise ValueError(
            f"Unable to identify framework of the model, please provide framework kwarg"
        )

    if framework in ("tf", "tensorflow", "keras"):
        from .tf_explanation_func import tf_explain

        return tf_explain(model, x_batch, y_batch, *args, method=method, **kwargs)
    if framework in ("torch", "pytorch", "pt"):
        from .torch_explanation_func import torch_explain

        return torch_explain(model, x_batch, y_batch, *args, method=method, **kwargs)

    raise ValueError(f"Unknown DNN framework {framework}")
