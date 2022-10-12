"""This module contains the utils functions of the library."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import re
import copy
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from importlib import util
from skimage.segmentation import slic, felzenszwalb
from ..helpers.model_interface import ModelInterface
from ..helpers import asserts

if util.find_spec("torch"):
    import torch
    from ..helpers.pytorch_model import PyTorchModel
if util.find_spec("tensorflow"):
    import tensorflow as tf
    from ..helpers.tf_model import TensorFlowModel


def get_superpixel_segments(img: np.ndarray, segmentation_method: str) -> np.ndarray:
    """
    Given an image, return segments or so-called 'super-pixels' segments i.e., an 2D mask with segment labels.

    Parameters
    ----------
    img: np.ndarray
            CxWxH image array.
    segmentation_method: string
        Indicates the segmentation method, i.e. "slic" or "felzenszwalb".

    Returns
    -------
    img: np.ndarray
        CxWxH segmented image array.
    """

    if img.ndim != 3:
        raise ValueError(
            "Make sure that x is 3 dimensional e.g., (3, 224, 224) to calculate super-pixels."
            f" shape: {img.shape}"
        )
    if segmentation_method not in ["slic", "felzenszwalb"]:
        raise ValueError(
            "'segmentation_method' must be either 'slic' or 'felzenszwalb'."
        )

    if segmentation_method == "slic":
        return slic(img, start_label=0)
    elif segmentation_method == "felzenszwalb":
        return felzenszwalb(
            img,
        )


def get_baseline_value(
    value: Union[float, int, str, np.array],
    arr: np.ndarray,
    return_shape: Tuple,
    patch: Optional[np.ndarray] = None,
    **kwargs,
) -> np.array:
    """
    Get the baseline value to fill the array with, in the shape of return_shape.

    Parameters
    ----------
    value: float, int, str, np.ndarray
        Either the value (float, int) to fill the array with, a method (string) used to construct
        baseline array ("mean", "uniform", "black", "white", "neighbourhood_mean" or
        "neighbourhood_random_min_max"), or the array (np.array) to be returned.
    arr: np.ndarray
        CxWxH image array used to calculate baseline values, i.e. for "mean", "black" and "white" methods.
    return_shape: tuple
        CxWxH shape to be returned.
    patch: np.ndarray, optional
        CxWxH patch array to calculate baseline values.
        Necessary for "neighbourhood_mean" and "neighbourhood_random_min_max" methods.
    kwargs: optional
            Keyword arguments.

    Returns
    -------
    np.ndarray
        Baseline array in return_shape.

    """

    kwargs["return_shape"] = return_shape
    if isinstance(value, (float, int)):
        return np.full(return_shape, value)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return np.full(return_shape, value)
        elif value.shape == return_shape:
            return value
        else:
            raise ValueError(
                "Shape {} of argument 'value' cannot be fitted to required shape {} of return value".format(
                    value.shape, return_shape
                )
            )
    elif isinstance(value, str):
        fill_dict = get_baseline_dict(arr, patch, **kwargs)
        if value.lower() == "random":
            raise ValueError(
                "'random' as a choice for 'perturb_baseline' is deprecated and has been removed from "
                "the current release. Please use 'uniform' instead and pass lower- and upper bounds to "
                "kwargs as see fit (default values are set to 'uniform_low=0.0' and 'uniform_high=1.0' "
                "which will replicate the results of 'random').\n"
            )
        if value.lower() not in fill_dict:
            raise ValueError(
                f"Ensure that 'value'(string) is in {list(fill_dict.keys())}"
            )
        return np.full(return_shape, fill_dict[value.lower()])
    else:
        raise ValueError("Specify 'value' as a np.array, string, integer or float.")


def get_baseline_dict(
    arr: np.ndarray, patch: Optional[np.ndarray] = None, **kwargs
) -> dict:
    """
    Make a dictionary of baseline approaches depending on the input x (or patch of input).

    Parameters
    ----------
    arr: np.ndarray
        CxWxH image array used to calculate baseline values, i.e. for "mean", "black" and "white" methods.
    patch: np.ndarray, optional
        CxWxH patch array to calculate baseline values, necessary for "neighbourhood_mean" and
        "neighbourhood_random_min_max" methods.
    kwargs: optional
            Keyword arguments..

    Returns
    -------
        fill_dict: dict
            Maps all available baseline methods to baseline values.
    """
    fill_dict = {
        "mean": float(arr.mean()),
        "uniform": np.random.uniform(
            low=kwargs.get("uniform_low", 0.0),
            high=kwargs.get("uniform_high", 1.0),
            size=kwargs["return_shape"],
        ),
        "black": float(arr.min()),
        "white": float(arr.max()),
    }
    if patch is not None:
        fill_dict["neighbourhood_mean"] = (float(patch.mean()),)
        fill_dict["neighbourhood_random_min_max"] = float(
            np.random.uniform(low=patch.min(), high=patch.max())
        )
    return fill_dict


def get_name(name: str):
    """
    Get the name of the Metric class object.

    Parameters
    ----------
    name: string
        A metric name.

    Returns
    -------
    name: string
        A cleaned version of the Metric name.
    """

    if name.isupper():
        return name
    return " ".join(re.sub(r"([A-Z])", r" \1", name).split())


def get_features_in_step(max_steps_per_input: int, input_shape: Tuple[int, ...]):
    """
    Get the number of features in the iteration.

    Parameters
    ----------
    max_steps_per_input: integer
        The number of repeated iterations on an image.
    input_shape: tuple
        Input shape.

    Returns
    -------
    float
        Product of the input shape divided by the maximum number of steps.
    """
    return int(np.prod(input_shape) / max_steps_per_input)


def filter_compatible_patch_sizes(perturb_patch_sizes: list, img_size: int) -> list:
    """
    Remove patch sizes that are not compatible with input size.

    Parameters
    ----------
    perturb_patch_sizes: list:
        Patch sizes for perturbation.
    img_size: integer
        A single dimension of an image array.

    Returns
    -------
    list:
        All integers within perturb_patch_sizes which are compatible with the image.
    """
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def infer_channel_first(x: np.array) -> bool:
    """
    Infer if the channels are first.

    Assumes:
        For 1d input:
            nr_channels < sequence_length

        For 2d input:
            nr_channels < img_width and nr_channels < img_height

    For higher dimensional input an error is raised.

    Parameters
    ----------
    x: np.ndarray
         Input image.

    Returns
    -------
    For 1d input:
        True if input shape is (nr_batch, nr_channels, sequence_length).
        False if input shape is (nr_batch, sequence_length, nr_channels).
        An error is raised if the two last dimensions are equal.

    For 2d input:
        True if input shape is (nr_batch, nr_channels, img_width, img_height).
        False if input shape is (nr_batch, img_width, img_height, nr_channels).
        An error is raised if the three last dimensions are equal.
    """
    err_msg = "Ambiguous input shape. Cannot infer channel-first/channel-last order."

    if len(np.shape(x)) == 3:
        if np.shape(x)[-2] < np.shape(x)[-1]:
            return True
        elif np.shape(x)[-2] > np.shape(x)[-1]:
            return False
        else:
            raise ValueError(err_msg)

    elif len(np.shape(x)) == 4:
        if np.shape(x)[-1] < np.shape(x)[-2] and np.shape(x)[-1] < np.shape(x)[-3]:
            return False
        if np.shape(x)[-3] < np.shape(x)[-1] and np.shape(x)[-3] < np.shape(x)[-2]:
            return True
        raise ValueError(err_msg)

    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def make_channel_first(x: np.array, channel_first=False):
    """
    Reshape batch to channel first.

    Parameters
    ----------
    x: np.ndarray
         The input image.

    Returns
    -------
    np.ndarray
        Image in CxWxH format.
    """
    if channel_first:
        return x

    if len(np.shape(x)) == 4:
        return np.moveaxis(x, -1, -3)
    elif len(np.shape(x)) == 3:
        return np.moveaxis(x, -1, -2)
    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def make_channel_last(x: np.array, channel_first=True):
    """
    Reshape batch to channel last.

    Parameters
    ----------
    x: np.ndarray
         The input image.

    Returns
    -------
    np.ndarray
        Image in WxHxC format.
    """
    if not channel_first:
        return x

    if len(np.shape(x)) == 4:
        return np.moveaxis(x, -3, -1)
    elif len(np.shape(x)) == 3:
        return np.moveaxis(x, -2, -1)
    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )


def get_wrapped_model(
    model,
    channel_first: bool,
    softmax: bool,
    device: Optional[str] = None,
    model_predict_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelInterface:
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.

    Parameters
    ----------
    model: Union[torch.nn.Module, tf.keras.Model]
        A model this will be wrapped in the ModelInterface:
    channel_first: boolean, optional
         Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
    softmax: boolean
        Indicates whether to use softmax probabilities or logits in model prediction. This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    model_predict_kwargs: dict, optional
        Keyword arguments to be passed to the model's predict method.

    Returns
    -------
    model: ModelInterface
        A wrapped ModelInterface model.
    """
    if util.find_spec("tensorflow"):
        if isinstance(model, tf.keras.Model):
            return TensorFlowModel(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                model_predict_kwargs=model_predict_kwargs,
            )
    if util.find_spec("torch"):
        if isinstance(model, torch.nn.Module):
            return PyTorchModel(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )
    raise ValueError("Model needs to be tf.keras.Model or torch.nn.Module.")


def blur_at_indices(
    arr: np.array,
    kernel: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
) -> np.array:
    """
    Creates a version of arr that is blurred at indices.

    Parameters
    ----------
    arr: np.ndarray
         Array to be perturbed.
    kernel: np.ndarray
         Kernel used for blurring.
    indices: int, sequence, tuple
        Array-like, with a subset shape of arr.
    indexed_axes: sequence
        The dimensions of arr that are indexed. These need to be consecutive and either
         include the first or last dimension of array.

    Returns
    -------
    np.ndarray
        A version of arr that is blurred at indices.
    """

    assert kernel.ndim == len(
        indexed_axes
    ), "kernel should have as many dimensions as indexed_axes has elements."

    # Pad array
    pad_width = [(0, 0) for _ in indexed_axes]
    for i, ax in enumerate(indexed_axes):
        pad_left = kernel.shape[i] // 2
        pad_right = kernel.shape[i] // 2 - (kernel.shape[i] % 2 == 0)
        pad_width[i] = (pad_left, pad_right)
    x = _pad_array(arr, pad_width, mode="constant", padded_axes=indexed_axes)

    # Handle indices
    indices = expand_indices(arr, indices, indexed_axes)
    none_slices = []
    array_indices = []
    for i, idx in enumerate(indices):
        if isinstance(idx, slice) and idx == slice(None):
            none_slices.append(idx)
        elif isinstance(idx, np.ndarray):
            pad_left = kernel.shape[[p for p in indexed_axes].index(i)] // 2
            array_indices.append(idx + pad_left)
        else:
            raise ValueError("Invalid indices {}".format(indices))
    array_indices = np.array(array_indices)

    # Expand kernel dimensions
    expanded_kernel = np.expand_dims(
        kernel, tuple([i for i in range(arr.ndim) if i not in indexed_axes])
    )

    # Iterate over indices, applying expanded kernel
    x_blur = copy.copy(x)
    for i in range(array_indices.shape[-1]):
        idx = list(array_indices[..., [i]])
        expanded_idx = copy.copy(idx)
        for ax, idx_ax in enumerate(expanded_idx):
            s = kernel.shape[ax]
            idx_ax = np.squeeze(idx_ax)
            expanded_idx[ax] = slice(
                idx_ax - (s // 2), idx_ax + s // 2 + 1 - (s % 2 == 0)
            )

        if 0 not in indexed_axes:
            expanded_idx = none_slices + expanded_idx
            idx = none_slices + idx
        expanded_idx = tuple(expanded_idx)
        idx = tuple(idx)

        x_blur[idx] = np.sum(
            np.multiply(x[expanded_idx], expanded_kernel),
            axis=tuple(indexed_axes),
            keepdims=True,
        )

    return _unpad_array(x_blur, pad_width, padded_axes=indexed_axes)


def create_patch_slice(
    patch_size: Union[int, Sequence[int]], coords: Sequence[int]
) -> Tuple[slice]:
    """
    Create a patch slice from patch size and coordinates.

    Parameters
    ----------
    patch_size: int, sequence
        One- or multidimensional patch size.
    coords: sequence
        Coordinates for creating patches.

    Returns
    -------
    tuple
        Patches at all provided coordinates.
    """

    if isinstance(patch_size, int):
        patch_size = (patch_size,)
    if isinstance(coords, int):
        coords = (coords,)

    patch_size = np.array(patch_size)
    coords = tuple(coords)

    if len(patch_size) == 1 and len(coords) != 1:
        patch_size = tuple(patch_size for _ in coords)
    elif patch_size.ndim != 1:
        raise ValueError("patch_size has to be either a scalar or a 1d-sequence")
    elif len(patch_size) != len(coords):
        raise ValueError(
            "patch_size sequence length does not match coords length"
            f" (len(patch_size) != len(coords))"
        )
    # make sure that each element in tuple is integer
    patch_size = tuple(int(patch_size_dim) for patch_size_dim in patch_size)

    patch_slice = [
        slice(coord, coord + patch_size_dim)
        for coord, patch_size_dim in zip(coords, patch_size)
    ]

    return tuple(patch_slice)


def get_nr_patches(
    patch_size: Union[int, Sequence[int]], shape: Tuple[int, ...], overlap: bool = False
) -> int:
    """
    Get number of patches for given shape.

    Parameters
    ----------
        patch_size: int, sequence
            One- or multidimensional patch size.
        shape (shape: Tuple[int, ...]): The image shape.
        overlap: boolean
            Indicates whether overlapping patches is used or not.

    Returns
    -------
    integer
        Number of patches that fit into the image.
    """

    if isinstance(patch_size, int):
        patch_size = (patch_size,)
    patch_size = np.array(patch_size)

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

    return np.prod(shape) // np.prod(patch_size)


def _pad_array(
    arr: np.array,
    pad_width: Union[int, Sequence[int], Sequence[Tuple[int]]],
    mode: str,
    padded_axes: Sequence[int],
) -> np.array:
    """
    Pad an array according to the mode.

    Parameters
    ----------
    arr: np.ndarray
         Array to pad.
    pad_width: int, sequence
            The size of the padding.
    mode: string
        numpy.pad mode (see https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
    padded_axes: sequence
        Indices of axes to pad.

    Returns
    -------
    np.ndarray
        Padded array.
    """

    assert (
        len(padded_axes) <= arr.ndim
    ), "Cannot pad more axes than array has dimensions"

    if isinstance(pad_width, Sequence):
        assert len(pad_width) == len(
            padded_axes
        ), "pad_width and padded_axes have different lengths"
        for p in pad_width:
            if isinstance(p, Tuple):
                assert len(p) == 2, "Elements in pad_width need to have length 2"

    pad_width_list = []

    for ax in range(arr.ndim):
        if ax not in padded_axes:
            pad_width_list.append((0, 0))
        elif isinstance(pad_width, int):
            pad_width_list.append((pad_width, pad_width))
        elif isinstance(pad_width[[p for p in padded_axes].index(ax)], int):
            pad_width_list.append(
                (
                    pad_width[[p for p in padded_axes].index(ax)],
                    pad_width[[p for p in padded_axes].index(ax)],
                )
            )
        else:
            pad_width_list.append(pad_width[[p for p in padded_axes].index(ax)])

    arr_pad = np.pad(arr, pad_width_list, mode=mode)

    return arr_pad


def _unpad_array(
    arr: np.array,
    pad_width: Union[int, Sequence[int], Sequence[Tuple[int]]],
    padded_axes: Sequence[int],
):
    """
    Remove padding from the array.

    Parameters
    ----------
    arr: np.ndarray
         A numpy array of the input to be unpaded.
    pad_witdh: int, sequence
        The width of the padding for the different dimensions.
    padded_axes: sequence
        The axes for padding.

    Returns
    -------
        np.ndarray:
        The unpadded array.

    """

    assert (
        len(padded_axes) <= arr.ndim
    ), "Cannot unpad more axes than array has dimensions"

    if isinstance(pad_width, Sequence):
        assert len(pad_width) == len(
            padded_axes
        ), "pad_width and padded_axes have different lengths"
        for p in pad_width:
            if isinstance(p, Tuple):
                assert len(p) == 2, "Elements in pad_width need to have length 2"

    unpad_slice = []
    for ax in range(arr.ndim):
        if ax not in padded_axes:
            unpad_slice.append(slice(None))
        elif isinstance(pad_width, int):
            unpad_slice.append(slice(pad_width, arr.shape[ax] - pad_width))
        elif isinstance(pad_width[[p for p in padded_axes].index(ax)], int):
            unpad_slice.append(
                slice(
                    pad_width[[p for p in padded_axes].index(ax)],
                    arr.shape[ax] - pad_width[[p for p in padded_axes].index(ax)],
                )
            )
        else:
            unpad_slice.append(
                slice(
                    pad_width[[p for p in padded_axes].index(ax)][0],
                    arr.shape[ax] - pad_width[[p for p in padded_axes].index(ax)][1],
                )
            )
    return arr[tuple(unpad_slice)]


def expand_attribution_channel(a_batch: np.ndarray, x_batch: np.ndarray):
    """
    Expand additional channel dimension(s) for attributions if needed.

    Parameters
    ----------
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    a_batch: np.ndarray
        An array which contains pre-computed attributions i.e., explanations.

    Returns
    -------
    np.ndarray
        A x_batch with dimensions matching those of a_batch.
    """
    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )
    if a_batch.ndim > x_batch.ndim:
        raise ValueError(
            f"a must not have greater ndim than x ({a_batch.ndim} > {x_batch.ndim})"
        )

    if a_batch.ndim == x_batch.ndim:
        return a_batch
    else:
        attr_axes = infer_attribution_axes(a_batch, x_batch)

        # TODO: Infer_attribution_axes currently returns dimensions w/o batch dimension.
        attr_axes = [a + 1 for a in attr_axes]
        expand_axes = [a for a in range(1, x_batch.ndim) if a not in attr_axes]

        return np.expand_dims(a_batch, axis=tuple(expand_axes))


def infer_attribution_axes(a_batch: np.ndarray, x_batch: np.ndarray) -> Sequence[int]:
    """
    Infers the axes in x_batch that are covered by a_batch.

    Parameters
    ----------
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    a_batch: np.ndarray
        An array which contains pre-computed attributions i.e., explanations.

    Returns
    -------
    np.ndarray
        The axes inferred.
    """
    # TODO: Adapt for batched processing.

    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )

    if a_batch.ndim > x_batch.ndim:
        raise ValueError(
            "Attributions need to have <= dimensions than inputs, but {} > {}".format(
                a_batch.ndim, x_batch.ndim
            )
        )

    # TODO: We currently assume here that the batch axis is not carried into the perturbation functions.
    a_shape = [s for s in np.shape(a_batch)[1:] if s != 1]
    x_shape = [s for s in np.shape(x_batch)[1:]]

    if a_shape == x_shape:
        return np.arange(0, len(x_shape))

    # One attribution value per sample
    if len(a_shape) == 0:
        return np.array([])

    x_subshapes = [
        [x_shape[i] for i in range(start, start + len(a_shape))]
        for start in range(0, len(x_shape) - len(a_shape) + 1)
    ]
    if x_subshapes.count(a_shape) < 1:

        # Check that attribution dimensions are (consecutive) subdimensions of inputs
        raise ValueError(
            "Attribution dimensions are not (consecutive) subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}".format(
                x_batch.shape, a_batch.shape
            )
        )
    elif x_subshapes.count(a_shape) > 1:

        # Check that attribution dimensions are (unique) subdimensions of inputs.
        # Consider potentially expanded dims in attributions.

        if a_batch.ndim == x_batch.ndim and len(a_shape) < a_batch.ndim:
            a_subshapes = [
                [np.shape(a_batch)[1:][i] for i in range(start, start + len(a_shape))]
                for start in range(0, len(np.shape(a_batch)[1:]) - len(a_shape) + 1)
            ]
            if a_subshapes.count(a_shape) == 1:

                # Inferring channel shape.
                for dim in range(len(np.shape(a_batch)[1:]) + 1):
                    if a_shape == np.shape(a_batch)[1:][dim:]:
                        return np.arange(dim, len(np.shape(a_batch)[1:]))
                    if a_shape == np.shape(a_batch)[1:][:dim]:
                        return np.arange(0, dim)

            raise ValueError(
                "Attribution axes could not be inferred for inputs of "
                "shape {} and attributions of shape {}".format(
                    x_batch.shape, a_batch.shape
                )
            )

        raise ValueError(
            "Attribution dimensions are not unique subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}."
            "Please expand attribution dimensions for a unique solution".format(
                x_batch.shape, a_batch.shape
            )
        )
    else:
        # Infer attribution axes.
        for dim in range(len(x_shape) + 1):
            if a_shape == x_shape[dim:]:
                return np.arange(dim, len(x_shape))
            if a_shape == x_shape[:dim]:
                return np.arange(0, dim)

    raise ValueError(
        "Attribution axes could not be inferred for inputs of "
        "shape {} and attributions of shape {}".format(x_batch.shape, a_batch.shape)
    )


def expand_indices(
    arr: np.array,
    indices: Union[int, Sequence[int], Tuple[np.array], Tuple[slice]],
    indexed_axes: Sequence[int],
) -> Tuple:
    """
    Expands indices to fit array shape. Returns expanded indices.
        --> if indices are a sequence of ints, they are interpreted as indices to the flattened arr,
            and subsequently expanded
        --> if indices contains only slices and 1d sequences for arr, everything is interpreted as slices
        --> if indices contains already expanded indices, they are returned as is

    Parameters
    ----------
    arr: np.ndarray
        The input to the expanded.
    indices: int, sequence, tuple
        A list of indices.
    indexed_axes: sequence
        Refers to all axes that are not indexed by slice(None).

    Returns
    -------
    tuple
        Expanded indices.
    """
    # TODO: Adapt for batched processing.

    # Handle indexed_axes.
    indexed_axes = np.sort(np.array(indexed_axes))
    asserts.assert_indexed_axes(arr, indexed_axes)

    # Handle indices.
    if isinstance(indices, int):
        expanded_indices = [indices]
    else:
        expanded_indices = []
        for i, idx in enumerate(indices):
            if isinstance(idx, slice) and idx == slice(None):
                pass
            elif isinstance(idx, slice):
                start = idx.start
                end = idx.stop
                step = idx.step
                tmp = np.arange(start, end, step)
                expanded_indices.append(tmp)
            elif isinstance(idx, np.ndarray):
                expanded_indices.append(idx)
            else:
                try:
                    expanded_indices.append(int(idx))
                except:
                    raise ValueError("Unsupported type of indices.")

    # Check if unraveling is needed.
    if np.all([isinstance(idx, int) for idx in expanded_indices]):
        expanded_indices = np.unravel_index(
            expanded_indices, tuple([arr.shape[i] for i in indexed_axes])
        )
    elif not np.all(
        [
            isinstance(idx, np.ndarray) and idx.ndim == len(expanded_indices)
            for idx in expanded_indices
        ]
    ):
        # Meshgrid sliced axes to account for correct slicing. Correct switched first two axes by meshgrid
        expanded_indices = [
            np.swapaxes(idx, 0, 1) if idx.ndim > 1 else idx
            for idx in np.meshgrid(*expanded_indices)
        ]

    # Handle case of 1D indices.
    if np.all([isinstance(idx, int) for idx in expanded_indices]):
        expanded_indices = [np.array(expanded_indices)]

    # Cast to list so item assignment works.
    expanded_indices = list(expanded_indices)

    if indexed_axes.size != len(expanded_indices):
        raise ValueError("indices dimension doesn't match indexed_axes")

    # Ensure array dimensions are kept when indexing.
    # Expands dimensions of each element in expanded_indices depending on the number of elements.
    for i in range(len(expanded_indices)):
        if expanded_indices[i].ndim != len(expanded_indices):
            expanded_indices[i] = np.expand_dims(
                expanded_indices[i],
                axis=tuple(range(len(expanded_indices) - expanded_indices[i].ndim)),
            )

    # Buffer with None-slices if indices index the last axes.
    for i in range(0, indexed_axes[0]):
        expanded_indices = slice(None), *expanded_indices

    return tuple(expanded_indices)


def get_leftover_shape(arr: np.array, axes: Sequence[int]) -> Tuple:
    """
    Gets the shape of the arr dimensions not included in axes.

    Parameters
    ----------
    arr: np.ndarray
         The input to the expanded.
    axes: sequence
        A sequence of ints containing the axes.

    Returns
    -------
    leftover_shape: tuple
        The leftover shape.
    """

    # TODO: Adapt for batched processing.
    axes = np.sort(np.array(axes))
    asserts.assert_indexed_axes(arr, axes)

    leftover_shape = tuple([arr.shape[i] for i in range(arr.ndim) if i not in axes])
    return leftover_shape


def offset_coordinates(indices: list, offset: tuple, img_shape: tuple):
    """
    Checks if offset coordinates are within the image frame.
        Adapted from: https://github.com/tleemann/road_evaluation.

    Parameters
    ----------
    indices: list
        List of indices to be offset.
    offset: tuple
        Offset for the coordinates, e.g. offset (1,1) adds 1 to both coordinates.
    img_shape: tuple
        Image shape in (channels, height, width) format.

    Returns
    -------
    list
        Offset coordinates for valid indices and the list of booleans which identifies valid ids.
    """
    x = indices // img_shape[2]
    y = indices % img_shape[2]
    x += offset[0]
    y += offset[1]
    valid = ~((x < 0) | (y < 0) | (x >= img_shape[1]) | (y >= img_shape[2]))
    off_coords = indices + offset[0] * img_shape[2] + offset[1]
    return off_coords[valid], valid


def calculate_auc(values: np.array, dx: int = 1.0):
    """
    Calculate area under the curve using the composite trapezoidal rule.

    Parameters
    ----------
    values: np.ndarray
         Input array.
    dx: integer
        The spacing between sample points. The default is 1.

    Returns
    -------
    np.ndarray
        Definite integral of values.
    """
    return np.trapz(np.array(values), dx=dx)
