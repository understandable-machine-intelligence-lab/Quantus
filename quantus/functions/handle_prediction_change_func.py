from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from quantus.helpers.model.model_interface import ModelInterface


def raise_exception_on_prediction_change(
    model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray
):
    """
    Raises ValueError in case prediction changed after applying perturbation to x_batch.

    Parameters
    ----------
    model: quantus.ModelInterface
        Model used to predict labels.
    x_batch: np.ndarray
        Batch of images.
    x_batch_perturbed: np.ndarray
        Batched of perturbed images.

    Returns
    -------

    x_batch_perturbed: np.ndarray
        Batched of perturbed images, if prediction didn't change, otherwise raise exception.

    """
    is_batched = len(x_batch.shape) == 4
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    if len(indexes_of_changed_labels) != 0:
        raise ValueError("Perturbation caused change in prediction.")
    return x_batch_perturbed if is_batched else x_batch_perturbed[0]


def warn_on_prediction_change(
    model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray
) -> np.ndarray:
    """
    Prints warning in case prediction changed after applying perturbation to x_batch.

    Parameters
    ----------
    model: quantus.ModelInterface
        Model used to predict labels.
    x_batch: np.ndarray
        Batch of images.
    x_batch_perturbed: np.ndarray
        Batched of perturbed images.

    Returns
    -------

    x_batch_perturbed: np.ndarray
        Batched of perturbed images.

    """
    is_batched = len(x_batch.shape) == 4
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    if len(indexes_of_changed_labels) != 0:
        warnings.warn(
            f"Perturbation caused change in predictions for indexes {indexes_of_changed_labels}"
        )
    return x_batch_perturbed if is_batched else x_batch_perturbed[0]


def return_nan_on_prediction_change(
    model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray
) -> np.ndarray:
    """
    Replaces images for which predictions changed with nan's.

    Parameters
    ----------
    model: quantus.ModelInterface
        Model used to predict labels.
    x_batch: np.ndarray
        Batch of images.
    x_batch_perturbed: np.ndarray
        Batched of perturbed images.

    Returns
    -------

    x_batch_perturbed: np.ndarray
        Batched of perturbed images, where images for which prediction changes are replaced by nan's.

    """
    is_batched = len(x_batch.shape) == 4
    img_shape = x_batch.shape[1:]
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    for i in indexes_of_changed_labels:
        x_batch_perturbed[i] = np.full(fill_value=np.nan, shape=img_shape)
    return x_batch_perturbed if is_batched else x_batch_perturbed[0]


def ignore_change_in_predictions(
    model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray
) -> np.ndarray:
    """
    Does nothing.

    Parameters
    ----------
    model: quantus.ModelInterface
        Unused.
    x_batch: np.ndarray
        Unused.
    x_batch_perturbed: np.ndarray
        Unused.

    Returns
    -------

    x_batch_perturbed: np.ndarray
        Batched of perturbed images.

    """
    return x_batch_perturbed
