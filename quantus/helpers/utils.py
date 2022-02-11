"""This module contains the utils functions of the library."""
import re
import random
import numpy as np
from typing import Union, Optional, List, Callable
from importlib import util
from skimage.segmentation import *
from ..helpers.model_interface import ModelInterface

if util.find_spec("torch"):
    import torch
    from ..helpers.pytorch_model import PyTorchModel
if util.find_spec("tensorflow"):
    import tensorflow as tf
    from ..helpers.tf_model import TensorFlowModel


def get_superpixel_segments(
    img: np.ndarray, segmentation_method: str, **kwargs
) -> np.ndarray:
    """Given an image, return segments or so-called 'super-pixels' segments i.e., an 2D mask with segment labels."""
    assert (
        len(img.shape) == 3
    ), "Make sure that x is 3 dimensional e.g., (3, 224, 224) to calculate super-pixels."
    assert segmentation_method in [
        "slic",
        "felzenszwalb",
    ], "Segmentation method must be either 'slic' or 'felzenszwalb'."

    if segmentation_method == "slic":
        return slic(img, start_label=0)
    elif segmentation_method == "felzenszwalb":
        return felzenszwalb(
            img,
        )
    else:
        print(
            "Segmentation method i.e., 'segmentation_method' must be either 'slic' or 'felzenszwalb'."
        )


def get_baseline_value(
    choice: Union[float, int, str, None], img: np.ndarray, **kwargs
) -> float:
    """Get the baseline value (float) to fill tensor with."""

    if choice is None:
        assert (
            ("perturb_baseline" in kwargs)
            or ("fixed_values" in kwargs)
            or ("constant_value" in kwargs)
            or ("input_shift" in kwargs)
        ), (
            "Specify"
            "a 'perturb_baseline', 'fixed_values', 'constant_value' or 'input_shift' e.g., 0.0 or 'black' for "
            "pixel replacement or 'baseline_values' containing an array with one value per index for replacement."
        )

    if "fixed_values" in kwargs:
        return kwargs["fixed_values"]
    if isinstance(choice, (float, int)):
        return choice
    elif isinstance(choice, str):
        fill_dict = get_baseline_dict(img, **kwargs)
        assert choice in list(
            fill_dict.keys()
        ), f"Ensure that 'perturb_baseline' or 'constant_value' (str) that exist in {list(fill_dict.keys())}"
        return fill_dict.get(choice.lower(), fill_dict["uniform"])
    else:
        raise print(
            "Specify 'perturb_baseline' or 'constant_value' as a string, integer or float."
        )


def get_baseline_dict(img: Union[np.ndarray, None], **kwargs) -> dict:
    """Make a dicionary of baseline approaches depending on the input x (or patch of input)."""
    fill_dict = {
        "mean": float(img.mean()),
        "random": float(random.random()),
        "uniform": float(random.uniform(img.min(), img.max())),
        "black": float(img.min()),
        "white": float(img.max()),
    }
    if "patch" in kwargs:
        fill_dict["neighbourhood_mean"] = (float(kwargs["patch"].mean()),)
        fill_dict["neighbourhood_random_min_max"] = (
            float(random.uniform(kwargs["patch"].min(), kwargs["patch"].max())),
        )
    return fill_dict


def get_name(str: str):
    """Get the name of the class object"""
    if str.isupper():
        return str
    return " ".join(re.sub(r"([A-Z])", r" \1", str).split())


def set_features_in_step(max_steps_per_input: int, img_size: int):
    return (img_size * img_size) / max_steps_per_input


def filter_compatible_patch_sizes(perturb_patch_sizes: list, img_size: int) -> list:
    """Remove patch sizes that are not compatible with input size."""
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def get_channel_first(x: np.array):
    """
    Returns True if input shape is (nr_batch, nr_channels, img_size, img_size).
    Returns False if input shape is (nr_batch, img_size, img_size, nr_channels).
    An error is raised if three last dimensions are equal, or if the image is not square.
    """
    if np.shape(x)[-1] == np.shape(x)[-2] == np.shape(x)[-3]:
        raise ValueError("Ambiguous input shape")
    if np.shape(x)[-3] == np.shape(x)[-2]:
        return False
    if np.shape(x)[-1] == np.shape(x)[-2]:
        return True
    raise ValueError("Input dimension mismatch")


def get_channel_first_batch(x: np.array, channel_first=False):
    """
    Reshape batch to channel first.
    """
    if channel_first:
        return x
    return np.moveaxis(x, -1, -3)


def get_channel_last_batch(x: np.array, channel_first=True):
    """
    Reshape batch to channel last.
    """
    if channel_first:
        return np.moveaxis(x, -3, -1)
    return x


def get_wrapped_model(model: ModelInterface, channel_first: bool) -> ModelInterface:
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.
    Return wrapped model.
    """
    if isinstance(model, tf.keras.Model):
        return TensorFlowModel(model, channel_first)
    if isinstance(model, torch.nn.modules.module.Module):
        return PyTorchModel(model, channel_first)
    raise ValueError(
        "Model needs to be tf.keras.Model or torch.nn.modules.module.Module."
    )
