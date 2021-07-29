import torch
from typing import Union, Optional
import numpy as np


def attr_check(method):
    pass
    #https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0
    """
    def attr_check(method):
        def inner(ref, name, age, service):
            for attr in [age, service]:
                if not isinstance(attr, int):
                    raise TypeError('age and service must be of type int')
            return method(ref, name, age, service)

        return inner
    """

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
