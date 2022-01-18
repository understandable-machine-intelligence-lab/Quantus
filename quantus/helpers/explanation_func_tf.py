"""This modules contains explainer functions which can be used in conjunction with the metrics in the library."""
from typing import Union

import numpy as np
import scipy
import random
import cv2
from tf_explain.core.vanilla_gradients import VanillaGradients
import warnings
from .utils import *
from .normalise_func import *
import tensorflow as tf


def explain_tf(
    model: tf.keras.Model,
    inputs: Union[np.array, torch.Tensor],
    targets: Union[np.array, torch.Tensor],
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Explain inputs given a model, targets and an explanation method.

    Expecting inputs to be shaped such as (batch_size, img_size, img_size, nr_channels)

    Returns np.ndarray of same shape as inputs.
    """

    if "method" not in kwargs:
        warnings.warn(
            f"Using quantus 'explain' function as an explainer without specifying 'method' (str) "
            f"in kwargs will produce a vanilla 'Gradient' explanation.\n",
            category=UserWarning,
        )

    method = kwargs.get("method", "Gradient").lower()

    inputs = inputs.reshape(
        -1,
        kwargs.get("img_size", 224),
        kwargs.get("img_size", 224),
        kwargs.get("nr_channels", 3),
    )

    explanation: np.ndarray = np.zeros_like(inputs)

    if method == "Gradient".lower():
        explanation = np.array(
            list(
                map(
                    lambda x, y: VanillaGradients().explain(([x], None), model, y),
                    inputs, targets
                )
            )
        )

    else:
        raise KeyError(
            "Specify a XAI method that already has been implemented {}."
        ).__format__("XAI_METHODS")

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    if kwargs.get("normalise", False):
        explanation = kwargs.get("normalise_func", normalise_by_negative)(explanation)

    if kwargs.get("abs", False):
        explanation = np.abs(explanation)

    elif kwargs.get("pos_only", False):
        explanation[explanation < 0] = 0.0

    elif kwargs.get("neg_only", False):
        explanation[explanation > 0] = 0.0

    return explanation
