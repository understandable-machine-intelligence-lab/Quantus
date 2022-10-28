from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from quantus.helpers.model.model_interface import ModelInterface


def raise_exception_on_prediction_change(model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray):
    is_batched = len(x_batch.shape) == 4
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    if len(indexes_of_changed_labels) != 0:
        raise ValueError("Perturbation caused change in prediction.")


def warn_on_prediction_change(model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray) -> np.ndarray:
    is_batched = len(x_batch.shape) == 4
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    if len(indexes_of_changed_labels) != 0:
        warnings.warn(f"Perturbation caused change in predictions for indexes {indexes_of_changed_labels}")
    return x_batch_perturbed if is_batched else x_batch_perturbed[0]


def return_nan_on_prediction_change(model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray) -> np.ndarray:
    is_batched = len(x_batch.shape) == 4
    if not is_batched:
        x_batch = np.expand_dims(x_batch, 0)
        x_batch_perturbed = np.expand_dims(x_batch_perturbed, 0)

    y_batch = model.predict(x_batch).argmax(axis=1)
    y_batch_perturbed = model.predict(x_batch_perturbed).argmax(axis=1)
    indexes_of_changed_labels = np.argwhere(y_batch != y_batch_perturbed).reshape(-1)
    for i in indexes_of_changed_labels:
        x_batch_perturbed[i] = np.nan
    return x_batch_perturbed if is_batched else x_batch_perturbed[0]


def ignore_change_in_predictions(model: ModelInterface, x_batch: np.ndarray, x_batch_perturbed: np.ndarray) -> np.ndarray:
    return x_batch_perturbed
