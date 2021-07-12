from typing import Union, List
import tensorflow as tf
import torch
from xai_quantification_toolbox.xai_quantification_toolbox.measures.base import Measure


def create_quantifier(
    model: Union[tf.keras.models.Model, torch.nn.Module], measures: List[Measure]
):
    if not hasattr(model, "fit"):
        raise TypeError(f"{model} is not an model instance.")
