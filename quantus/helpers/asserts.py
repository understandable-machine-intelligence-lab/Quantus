"""This module holds a collection of asserts functionality that is used across the Quantus library to avoid undefined behaviour."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Callable, Tuple, Sequence, Union

import numpy as np


def attributes_check(metric):
    """
    Basic check of all the metrics, passed as a decorator.

    Parameters
    ----------
    metric: base.Metric class
        The metric class.

    Returns
    -------
    base.Metric class
        The metric class.

    """
    attr = metric.__dict__
    if "abs" in attr:
        if not bool(attr["abs"]):
            raise TypeError("The 'abs' must be a bool.")
    if "normalise" in attr:
        assert bool(attr["normalise"]), "The 'normalise' must be a bool."
    if "return_aggregate" in attr:
        assert bool(attr["return_aggregate"]), "The 'return_aggregate' must be a bool."
    if "disable_warnings" in attr:
        assert bool(attr["disable_warnings"]), "The 'disable_warnings' must be a bool."
    if "display_progressbar" in attr:
        assert bool(
            attr["display_progressbar"]
        ), "The 'display_progressbar' must be a bool."
    return metric


def assert_features_in_step(
    features_in_step: int, input_shape: Tuple[int, ...]
) -> None:
    """
        Assert that features in step is compatible with the image size.

        Parameters
        ----------
        features_in_step: integer
            The number of features e.g., pixels included in each iteration.
        input_shape: Tuple[int...]
            The shape of the input.

        Returns
        -------
    None
    """
    assert np.prod(input_shape) % features_in_step == 0, (
        "Set 'features_in_step' so that the modulo remainder "
        "returns zero given the product of the input shape."
        f" ({np.prod(input_shape)} % {features_in_step} != 0)"
    )


def assert_patch_size(patch_size: Union[int, tuple], shape: Tuple[int, ...]) -> None:
    """
    Assert that patch size is compatible with given image shape.

    Parameters
    ----------
    patch_size: integer
        The size of the patch_size, assumed to tbe squared.
    input_shape: Tuple[int...]
        the shape of the input.

    Returns
    -------
    None
    """

    if isinstance(patch_size, int):
        patch_size = (patch_size,)
    patch_size = np.array(patch_size)

    assert isinstance(patch_size, np.ndarray)
    if len(patch_size) == 1 and len(shape) != 1:
        patch_size = tuple(patch_size for _ in shape)
    elif patch_size.ndim != 1:
        raise ValueError("patch_size has to be either a scalar or a 1d-sequence")
    elif len(patch_size) != len(shape):
        raise ValueError(
            "patch_size sequence length does not match shape length"
            f" (len(patch_size) != len(shape))"
        )
    patch_size = tuple(patch_size)
    if np.prod(shape) % np.prod(patch_size) != 0:
        raise ValueError(
            "Set 'patch_size' so that the input shape modulo remainder returns 0"
            f" [np.prod({shape}) % np.prod({patch_size}) != 0"
            f" => {np.prod(shape)} % {np.prod(patch_size)} != 0]"
        )


def assert_attributions_order(order: str) -> None:
    """
    Assert that order is in pre-defined list.

    Parameters
    ----------
    order: string
        The different orders that attributions could be ranked in.

    Returns
    -------
    None
    """
    assert order in [
        "random",
        "morf",
        "lorf",
    ], "The order of sorting the attributions must be either random, morf, or lorf."


def assert_nr_segments(nr_segments: int) -> None:
    """
    Assert that the number of segments given the segmentation algorithm is more than one.

    Parameters
    ----------
    nr_segments: integer
        The number of segments that the segmentaito algorithm produced.

    Returns
    -------
    None
    """
    assert (
        nr_segments > 1
    ), "The number of segments from the segmentation algorithm must be more than one."


def assert_layer_order(layer_order: str) -> None:
    """
    Assert that layer order is in pre-defined list.

    Parameters
    ----------
    layer_order: string
        The various ways that a model's weights of a layer can be randomised.

    Returns
    -------
    None
    """
    assert layer_order in ["top_down", "bottom_up", "independent"]


def assert_attributions(x_batch: np.array, a_batch: np.array) -> None:
    """
    Asserts on attributions, assumes channel first layout.

    Parameters
    ----------
    x_batch: np.ndarray
         The batch of input to compare the shape of the attributions with.
    a_batch: np.ndarray
         The batch of attributions.

    Returns
    -------
    None
    """
    assert (
        type(a_batch) == np.ndarray
    ), "Attributions 'a_batch' should be of type np.ndarray."
    assert np.shape(x_batch)[0] == np.shape(a_batch)[0], (
        "The inputs 'x_batch' and attributions 'a_batch' should "
        "include the same number of samples."
        "{} != {}".format(np.shape(x_batch)[0], np.shape(a_batch)[0])
    )
    assert np.ndim(x_batch) == np.ndim(a_batch), (
        "The inputs 'x_batch' and attributions 'a_batch' should "
        "have the same number of dimensions."
        "{} != {}".format(np.ndim(x_batch), np.ndim(a_batch))
    )
    a_shape = [s for s in np.shape(a_batch)[1:] if s != 1]
    x_shape = [s for s in np.shape(x_batch)[1:]]
    assert a_shape[0] == x_shape[0] or a_shape[-1] == x_shape[-1], (
        "The dimensions of attribution and input per sample should correspond in either "
        "the first or last dimensions, but got shapes "
        "{} and {}".format(a_shape, x_shape)
    )
    assert all([a in x_shape for a in a_shape]), (
        "All attribution dimensions should be included in the input dimensions, "
        "but got shapes {} and {}".format(a_shape, x_shape)
    )
    assert all(
        [
            x_shape.index(a) > x_shape.index(a_shape[i])
            for a in a_shape
            for i in range(a_shape.index(a))
        ]
    ), (
        "The dimensions of the attribution must correspond to dimensions of the input in the same order, "
        "but got shapes {} and {}".format(a_shape, x_shape)
    )
    assert not np.all((a_batch == 0)), (
        "The elements in the attribution vector are all equal to zero, "
        "which may cause inconsistent results since many metrics rely on ordering. "
        "Recompute the explanations."
    )
    assert not np.all((a_batch == 1.0)), (
        "The elements in the attribution vector are all equal to one, "
        "which may cause inconsistent results since many metrics rely on ordering. "
        "Recompute the explanations."
    )
    assert len(set(a_batch.flatten().tolist())) > 1, (
        "The attributions are uniformly distributed, "
        "which may cause inconsistent results since many "
        "metrics rely on ordering."
        "Recompute the explanations."
    )
    assert not np.all((a_batch < 0.0)), "Attributions should not all be less than zero."


def assert_segmentations(x_batch: np.array, s_batch: np.array) -> None:
    """
    Asserts on segmentations, assumes channel first layout.

    Parameters
    ----------
    x_batch: np.ndarray
         The batch of input to compare the shape of the attributions with.
    s_batch: np.ndarray
         The batch of segmentations.

    Returns
    -------
    None
    """
    assert (
        type(s_batch) == np.ndarray
    ), "Segmentations 's_batch' should be of type np.ndarray."
    assert (
        np.shape(x_batch)[0] == np.shape(s_batch)[0]
    ), "The inputs 'x_batch' and segmentations 's_batch' should include the same number of samples."
    assert (
        np.shape(x_batch)[2:] == np.shape(s_batch)[2:]
    ), "The inputs 'x_batch' and segmentations 's_batch' should share the same dimensions."
    assert (
        np.shape(s_batch)[1] == 1
    ), "The second dimension of the segmentations 's_batch' should be equal to 1."
    assert (
        len(np.nonzero(s_batch)) > 0
    ), "The segmentation 's_batch' must contain non-zero elements."
    assert (
        np.isin(s_batch.flatten(), [0, 1]).all()
        or np.isin(s_batch.flatten(), [True, False]).all()
    ), "The segmentation 's_batch' should contain only [1, 0] or [True, False]."


def assert_plot_func(plot_func: Callable) -> None:
    """
    Assert that the plot function is a callable.

    Parameters
    ----------
    plot_func: callable
        An plot function input, asusmed to be a Callable.

    Returns
    -------
    None
    """
    assert callable(plot_func), "Make sure that 'plot_func' is a callable."


def assert_explain_func(explain_func: Callable) -> None:
    """
    Asser thta the explanation function is a callable.

    Parameters
    ----------
    explain_func: callable
        An plot function input, asusmed to be a Callable.

    Returns
    -------
    None
    """
    assert callable(explain_func), (
        "Make sure 'explain_func' is a Callable that takes model, inputs, "
        "targets and **kwargs as arguments."
    )


def assert_value_smaller_than_input_size(
    x: np.ndarray, value: int, value_name: str
) -> None:
    """
    Checks if value is smaller than input size, assumes batch and channel first dimension.

    Parameters
    ----------
    x: np.ndarray
         The input to check the value against.
    value: integer
        The value that must be smaller than input size.
    value_name: string
        The hyperparameter to check, e.g., "k" for TopKIntersection.

    Returns
    -------
    None
    """
    if value >= np.prod(x.shape[2:]):
        raise ValueError(
            f"'{value_name}' must be smaller than input size."
            f" [{value} >= {np.prod(x.shape[2:])}]"
        )


def assert_indexed_axes(arr: np.array, indexed_axes: Sequence[int]) -> None:
    """
    Checks that indexed_axes fits the given array.

    Parameters
    ----------
    arr: np.ndarray
         A given array that we want to check indexed_axes against.
    indexed_axes: sequence
            The sequence with indices, with axes.

    Returns
    -------
    None
    """
    # TODO: Change for batching update, since currently single images are expected.
    assert len(indexed_axes) <= arr.ndim
    assert len(indexed_axes) == len(np.arange(indexed_axes[0], indexed_axes[-1] + 1))
    assert all(
        [
            a == b
            for a, b in list(
                zip(indexed_axes, np.arange(indexed_axes[0], indexed_axes[-1] + 1))
            )
        ]
    ), "Make sure indexed_axes contains consecutive axes."
    assert (
        0 in indexed_axes or arr.ndim - 1 in indexed_axes
    ), "Make sure indexed_axes contains either the first or last axis of arr."
