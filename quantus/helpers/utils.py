"""This module contains the utils functions of the library."""
import re
import torch
import random
from typing import Union, Optional, List, Callable
import numpy as np
from skimage.segmentation import *


def get_layers(model, order: str = "top_down"):
    """Checks a pytorch model for randomizable layers and returns them in a dict."""
    layers = [
        module
        for module in model.named_modules()
        if hasattr(module[1], "reset_parameters")
    ]

    if order == "top_down":
        return layers[::-1]
    else:
        return layers


def get_superpixel_segments(
    img: torch.Tensor, segmentation_method: str, **kwargs
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
        return slic(
            img,
            start_label=0,
        )

    elif segmentation_method == "felzenszwalb":
        return felzenszwalb(
            img,
        )
    else:
        print(
            "Segmentation method i.e., 'segmentation_method' must be either 'slic' or 'felzenszwalb'."
        )


def get_baseline_value(
    choice: Union[float, int, str, None], img: torch.Tensor, **kwargs
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


def get_baseline_dict(img: Union[torch.Tensor, None], **kwargs) -> dict:
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
