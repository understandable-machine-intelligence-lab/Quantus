from typing import Union
import numpy as np
import torch


def normalise_by_max(a: np.ndarray) -> np.ndarray:
    """ "Normalize attributions by the maximum absolute value of the explanation."""
    a /= np.max(np.abs(a))
    return a


def normalise_if_negative(a: np.ndarray) -> np.ndarray:
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""
    if a.min() >= 0.0:
        return a / a.max()
    if a.max() <= 0.0:
        return -a / a.min()
    return (a > 0.0) * a / a.max() - (a < 0.0) * a / a.min()


def denormalise_image(
    img,
    mean=np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1),
    std=np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1),
    **params,
) -> Union[np.ndarray, torch.Tensor]:
    """De-normalise a torch image (using conventional ImageNet values)."""
    if isinstance(img, torch.Tensor):
        return (
            img.view(
                [
                    params.get("nr_channels", 3),
                    params.get("img_size", 224),
                    params.get("img_size", 224),
                ]
            )
            * std
        ) + mean
    elif isinstance(img, np.ndarray):
        return (img * std) + mean
    else:
        print("Make image either a np.array or torch.Tensor before denormalising.")
        return img