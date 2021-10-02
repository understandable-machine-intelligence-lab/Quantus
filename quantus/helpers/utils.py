"""This module contains the utils functions of the library."""
import torch
import random
from typing import Union, Optional, List, Callable
import numpy as np
from skimage.segmentation import *


def get_layers(model, order: str = "top_down"):
    """ Checks a pytorch model for randomizable layers and returns them in a dict. """
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
            n_segments=kwargs.get("slic_n_segments", 224),
            compactness=kwargs.get("slic_compactness", 0.05),
            sigma=kwargs.get("slic_sigma", 0.1),
        )

    elif segmentation_method == "felzenszwalb":
        return felzenszwalb(
            img,
            scale=kwargs.get("felzenszwalb_scale", 448),
            sigma=kwargs.get("felzenszwalb_sigma", 0.1),
            min_size=kwargs.get("felzenszwalb_min_size", 112),
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
        ), (
            "Specify"
            "a 'perturb_baseline' or 'constant_value e.g., 0.0 or 'black' for pixel replacement or 'baseline_values' "
            "containing an array with one value per index for replacement."
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


def filter_compatible_patch_sizes(perturb_patch_sizes: list, img_size: int) -> list:
    """Remove patch sizes that are not compatible with input size."""
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def set_features_in_step(max_steps_per_input: int, img_size: int):
    return (img_size * img_size) / max_steps_per_input


def normalize_heatmap(heatmap: np.array):
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""
    if heatmap.min() >= 0.0:
        return heatmap / heatmap.max()
    if heatmap.max() <= 0.0:
        return -heatmap / heatmap.min()
    return (heatmap > 0.0) * heatmap / heatmap.max() - (
        heatmap < 0.0
    ) * heatmap / heatmap.min()


def denormalize_image(
    image,
    mean=np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1),
    std=np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1),
    **params,
):
    """De-normalize a torch image (using conventional ImageNet values)."""
    if isinstance(image, torch.Tensor):
        return (
            image.view(
                [
                    params.get("nr_channels", 3),
                    params.get("img_size", 224),
                    params.get("img_size", 224),
                ]
            )
            * std
        ) + mean
    elif isinstance(image, np.ndarray):
        std
        return (image * std) + mean
    else:
        print("Make image either a np.array or torch.Tensor before denormalizing.")
        return image


def check_if_fitted(m) -> Optional[bool]:
    """Checks if a measure is fitted by the presence of """
    if not hasattr(m, "fit"):
        raise TypeError(f"{m} is not an instance.")
    return True


"""
    if not check_if_fitted:
     print(f"This {Measure.name} is instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
"""
