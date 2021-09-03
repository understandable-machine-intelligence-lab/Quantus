import torch
import random
from typing import Union, Optional, List, Callable
import numpy as np
from skimage.segmentation import *


def get_superpixel_segments(img: torch.Tensor,
                    method: str,
                    **kwargs) -> np.ndarray:
    """Given an image, return segments or so-called 'super-pixels' segments i.e., an 2D mask with segment labels."""
    assert len(img.shape) == 3, "Make sure that x is 3 dimensional e.g., (3, 224, 224) to calculate super-pixels."
    assert method in ["slic", "felzenszwalb"], "Segmentation method must be either 'slic' or 'felzenszwalb'."

    if method == "slic":
        return slic(img,
                    n_segments=kwargs.get("slic_n_segments", 224),
                    compactness=kwargs.get("slic_compactness", 0.05),
                    sigma=kwargs.get("slic_sigma", 0.1))
    elif method == "felzenszwalb":
        return felzenszwalb(img,
                            scale=kwargs.get("felzenszwalb_scale", 448),
                            sigma=kwargs.get("felzenszwalb_sigma", 0.1),
                            min_size=kwargs.get("felzenszwalb_min_size", 112))
    else:
        print("Segmentation method i.e., 'segmentation_method' must be either 'slic' or 'felzenszwalb'.")


def get_baseline_value(choice: Union[float, int, str, None],
                       img: torch.Tensor,
                       **kwargs) -> float:
    """Get the baseline value (float) to fill tensor with."""

    if choice is None:
        assert ("perturb_baseline" in kwargs) or ("fixed_values" in kwargs) or ("constant_value" in kwargs), "Specify \
        a 'perturb_baseline' or 'constant_value e.g., 0.0 or 'black' for pixel replacement or 'baseline_values' containing \
        an array with one value per index for replacement."

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
        raise print("Specify 'perturb_baseline' or 'constant_value' as a string, integer or float.")


def get_baseline_dict(img: torch.Tensor, **kwargs) -> dict:
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


def assert_model_predictions_deviations(
    y_pred: torch.Tensor, y_pred_perturb: torch.Tensor, threshold: float = 0.01
):
    # TODO. Implement.
    pass


def assert_model_predictions_correct(
    y_pred: torch.Tensor, y_pred_perturb: torch.Tensor
):
    # TODO. Implement.
    pass


def attr_check(metric):
    # https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0
    attr = metric.__dict__
    if "perturb_func" in attr:
        if not callable(attr["perturb_func"]):
            raise TypeError("The 'perturb_func' must be a callable.")
    if "similarity_func" in attr:
        assert callable(attr["similarity_func"]), "The 'similarity_func' must be a callable."
    if "text_warning" in attr:
        assert isinstance(attr["text_warning"], str), "The 'text_warning' function must be a string."
    return metric


def set_warn(call):
    # TODO. Implement warning logic of decorator if text_warning is an attribute in class.
    def call_fn(*args):
        return call_fn
    return call
    #attr = call.__dict__
    #print(dir(call))
    #attr = {}
    #if "text_warning" in attr:
    #    call.print_warning(text=attr["text_warning"])
    #else:
    #    print("Do nothing.")
    #    pass


def assert_features_in_step(features_in_step: int,
                            img_size: int) -> None:
    """Assert that features in step is compatible with the image size."""
    assert (img_size * img_size) % features_in_step == 0, "Set 'features_in_step' so that the modulo remainder " \
                                                          "returns zero given the img_size."


def assert_max_steps(max_steps_per_input: int,
                     img_size: int) -> None:
    """Assert that max steps per inputs is compatible with the image size."""
    assert (img_size * img_size) % max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder " \
                                                          "returns zero given the img_size."


def assert_patch_size(patch_size: int, img_size: int) -> None:
    """Assert that patch size that are not compatible with input size."""
    assert (img_size % patch_size == 0), "Set 'patch_size' so that the modulo remainder returns 0 given the image size."


def assert_atts(a_batch: np.array,
                x_batch: np.array) -> None:
    """Asserts on attributions."""
    assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
    ), "Inputs and attributions should include the same number of samples."
    assert type(a_batch) == np.ndarray, "Attributions should be of type np.ndarray."


def assert_explain_func(explain_func: Callable) -> None:
    assert callable(explain_func), "Make sure 'explain_func' is a callable that takes model, x_batch, \
                            y_batch and **kwargs as arguments."


def filter_compatible_patch_sizes(perturb_patch_sizes: list,
                                   img_size: int) -> list:
    """Remove patch sizes that are not compatible with input size."""
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def set_features_in_step(max_steps_per_input: int,
                         img_size: int):
    return (img_size * img_size) / max_steps_per_input


def normalize_heatmap(heatmap: np.array):
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""
    # TODO. Debug this function so that it works with batches.
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
