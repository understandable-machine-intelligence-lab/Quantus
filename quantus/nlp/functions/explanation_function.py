import numpy as np
from typing import List, Dict


from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
    HuggingFaceTextClassifierTF,
)
from quantus.nlp.helpers.types import Explanation
from nlp.functions.tf_explanation_function import tf_explain
import tensorflow as tf


def explain(
    x_batch: List[str],
    y_batch: np.ndarray,
    model: TextClassifier,
    method: str,
    **kwargs: Dict,
) -> List[Explanation]:
    if isinstance(model, (tf.keras.Model, HuggingFaceTextClassifierTF)):
        return tf_explain(x_batch, y_batch, model, method=method, **kwargs)
    raise NotImplementedError()
