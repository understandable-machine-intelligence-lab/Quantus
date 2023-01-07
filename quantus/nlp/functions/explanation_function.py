import numpy as np
from typing import List, Dict


from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    HuggingFaceTextClassifierTF,
)
from quantus.nlp.helpers.types import Explanation
from nlp.functions.tf_explanation_function import tf_explain


def explain(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
    explain_fn_kwargs: Dict,
) -> List[Explanation]:
    if isinstance(model, HuggingFaceTextClassifierTF):
        return tf_explain(x_batch, y_batch, model, **explain_fn_kwargs)
    raise NotImplementedError()
