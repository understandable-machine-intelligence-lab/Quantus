"""Framework-agnostic explanation functions."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
import warnings
from transformers import pipeline
from functools import partial
from importlib import util

if util.find_spec("tensorflow"):
    from quantus.nlp.functions.tf_explanation_func import tf_explain

if util.find_spec("torch"):
    from quantus.nlp.functions.torch_explanation_func import torch_explain

from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import (
    value_or_default,
    safe_isinstance,
    add_default_items,
)

TF_HuggingfaceModelClass = "quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier.TensorFlowHuggingFaceTextClassifier"
TF_ModelClass = (
    "quantus.nlp.helpers.model.tensorflow_text_classifier.TensorflowTextClassifier"
)
Torch_HuggingfaceModelClass = "quantus.nlp.helpers.model.torch_huggingface_text_classifier.TorchHuggingFaceTextClassifier"
Torch_ModelClass = "quantus.nlp.helpers.model.torch_text_classifier.TorchTextClassifier"


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

    from lime.lime_text import LimeTextExplainer

    init_kwargs = add_default_items(init_kwargs, {"mask_string": "[MASK]"})
    call_kwargs = add_default_items(call_kwargs, {"top_labels": 1})
    predict_fn = partial(model.predict, batch_size=batch_size)

    explainer = LimeTextExplainer(**init_kwargs)

    explanations = []
    for x, y in zip(x_batch, y_batch):
        ex = explainer.explain_instance(x, predict_fn, **call_kwargs).as_list(label=y)
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

    import shap
    import shap.maskers

    init_kwargs = value_or_default(init_kwargs, lambda: {})
    call_kwargs = add_default_items(call_kwargs, {"silent": True})
    predict_fn = partial(model.predict, batch_size=batch_size)

    if safe_isinstance(model, (TF_HuggingfaceModelClass, Torch_HuggingfaceModelClass)):
        predict_fn = pipeline(
            "text-classification",
            model=model._model,  # type: ignore
            tokenizer=model._tokenizer._tokenizer,  # type: ignore
            top_k=None,
            device=getattr(model, "device", None),
        )
        explainer = shap.Explainer(predict_fn, **init_kwargs)
    else:
        explainer = shap.PartitionExplainer(
            predict_fn, shap.maskers.Text(), **init_kwargs  # noqa
        )

    shapley_values = explainer(x_batch, batch_size=batch_size, **call_kwargs)
    return [(i.feature_names, i.values[:, y]) for i, y in zip(shapley_values, y_batch)]


def _is_torch_model(model: TextClassifier):
    if safe_isinstance(model, (Torch_HuggingfaceModelClass, Torch_ModelClass)):
        return True
    return safe_isinstance(getattr(model, "internal_model", None), "torch.nn.Module")


def _is_tf_model(model: TextClassifier):
    if safe_isinstance(model, (TF_HuggingfaceModelClass, TF_ModelClass)):
        return True
    return safe_isinstance(
        getattr(model, "internal_model", None), ("keras.Model", "tensorflow.Module")
    )


def explain(
    model: TextClassifier,
    *args,
    method: Optional[str] = None,
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
        return explain_lime(model, *args, **kwargs)
    if method == "SHAP":
        return explain_shap(model, *args, **kwargs)

    if _is_tf_model(model):
        return tf_explain(model, *args, method=method, **kwargs)

    if _is_torch_model(model):
        return torch_explain(model, *args, method=method, **kwargs)

    raise ValueError(f"Unable to identify DNN framework of the model.")
