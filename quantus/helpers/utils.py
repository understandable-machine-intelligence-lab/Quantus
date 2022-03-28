"""This module contains the utils functions of the library."""
import re
import random
import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union
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
    """Given an image, return segments or so-called 'super-pixels' segments i.e., an 2D mask with segment labels."""
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
    """Get the baseline value to fill the array with, in the shape of return_shape"""

    if isinstance(value, (float, int)):
        return np.full(return_shape, value)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return np.full(return_shape, value)
        elif value.shape == return_shape:
            return value
        else:
            raise ValueError("Shape {} of argument 'value' cannot be fitted to required shape {} of return value".format(value.shape, return_shape))
    elif isinstance(value, str):
        fill_dict = get_baseline_dict(arr, patch)
        if value.lower() not in fill_dict:
            raise ValueError(
                f"Ensure that 'choice'(str) is in {list(fill_dict.keys())}"
            )
        return np.full(return_shape, fill_dict[value.lower()])
    else:
        raise ValueError(
            "Specify 'value'' as a np.array, string, integer or float."
        )


def get_baseline_dict(arr: np.ndarray, patch: Optional[np.ndarray] = None) -> dict:
    """Make a dicionary of baseline approaches depending on the input x (or patch of input)."""
    fill_dict = {
        "mean": float(arr.mean()),
        "random": float(random.random()),
        "uniform": float(random.uniform(arr.min(), arr.max())),
        "black": float(arr.min()),
        "white": float(arr.max()),
    }
    if patch is not None:
        fill_dict["neighbourhood_mean"] = float(patch.mean()),
        fill_dict["neighbourhood_random_min_max"] = float(random.uniform(patch.min(), patch.max()))
    return fill_dict


def get_name(str: str):
    """Get the name of the class object."""
    if str.isupper():
        return str
    return " ".join(re.sub(r"([A-Z])", r" \1", str).split())


def get_features_in_step(max_steps_per_input: int, input_shape: Tuple[int, ...]):
    """Get the number of features in the iteration."""
    return np.prod(input_shape) / max_steps_per_input


def filter_compatible_patch_sizes(perturb_patch_sizes: list, img_size: int) -> list:
    """Remove patch sizes that are not compatible with input size."""
    return [i for i in perturb_patch_sizes if img_size % i == 0]


def infer_channel_first(x: np.array):
    """
    Infer if the channels are first.

    For 1d input:

        Assumes
            nr_channels < sequence_length
        Returns
            True if input shape is (nr_batch, nr_channels, sequence_length).
            False if input shape is (nr_batch, sequence_length, nr_channels).
            An error is raised if the two last dimensions are equal.

    For 2d input:

        Assumes
            nr_channels < img_width and nr_channels < img_height
        Returns
            True if input shape is (nr_batch, nr_channels, img_width, img_height).
            False if input shape is (nr_batch, img_width, img_height, nr_channels).
            An error is raised if the three last dimensions are equal.

    For higher dimensional input an error is raised.
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
    """Reshape batch to channel first."""
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
    """Reshape batch to channel last."""
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


def get_wrapped_model(model: ModelInterface, channel_first: bool) -> ModelInterface:
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.

    Returns
        A wrapped ModelInterface model.
    """
    if isinstance(model, tf.keras.Model):
        return TensorFlowModel(model, channel_first)
    if isinstance(model, torch.nn.modules.module.Module):
        return PyTorchModel(model, channel_first)
    raise ValueError(
        "Model needs to be tf.keras.Model or torch.nn.modules.module.Module."
    )


def conv2D_numpy(
    x: np.array,
    kernel: np.array,
    stride: int,
    padding: int,
    groups: int,
    pad_output: bool = False,
) -> np.array:
    """
    Computes 2D convolution in NumPy.

    Assumes
        Shape of x is [C_in, H, W] with C_in = input channels and H, W input height and weight, respectively
        Shape of kernel is [C_out, C_in/groups, K, K] with C_out = output channels and K = kernel size
    """

    # Pad input
    x = np.pad(x, [(0, 0), (padding, padding), (padding, padding)], mode="constant")

    # Get shapes
    c_in, height, width = x.shape
    c_out, kernel_size = kernel.shape[0], kernel.shape[2]

    # Handle groups
    assert c_in % groups == 0
    assert c_out % groups == 0
    assert kernel.shape[1] * groups == c_in
    c_in_g = c_in // groups
    c_out_g = c_out // groups

    # Build output
    output_height = (height - kernel_size) // stride + 1
    output_width = (width - kernel_size) // stride + 1
    output = np.zeros((c_out, output_height, output_width)).astype(x.dtype)

    # TODO: improve efficiency, less loops
    for g in range(groups):
        for c in range(c_out_g * g, c_out_g * (g + 1)):
            for h in range(output_height):
                for w in range(output_width):
                    output[c][h][w] = np.multiply(
                        x[
                            c_in_g * g : c_in_g * (g + 1),
                            h * stride : h * stride + kernel_size,
                            w * stride : w * stride + kernel_size,
                        ],
                        kernel[c, :, :, :],
                    ).sum()

    if pad_output:
        if stride != 1 or padding != 0:
            raise NotImplementedError()
        padwidth = (kernel_size - 1) // 2
        output = np.pad(
            output,
            (
                (0, 0),
                (padwidth + (1 - (kernel_size % 2)), padwidth),
                (padwidth + (1 - (kernel_size % 2)), padwidth),
            ),
            mode="edge",
        )

    return output


def create_patch_slice(
    patch_size: Union[int, Sequence[int]], coords: Sequence[int]
) -> Tuple[Sequence[int]]:
    """
    Create a patch slice from patch size and coordinates.
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
        np.arange(coord, coord + patch_size_dim)
        for coord, patch_size_dim in zip(coords, patch_size)
    ]

    return tuple(patch_slice)

def get_nr_patches(
    patch_size: Union[int, Sequence[int]], shape: Tuple[int, ...], overlap: bool = False
) -> int:
    """Get number of patches for given shape."""
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


def _pad_array(arr: np.array, pad_width: int, mode: str, padded_axes: Sequence[int]):
    """To allow for any patch_size we add padding to the array."""

    assert len(padded_axes) <= arr.ndim, (
        "Cannot pad more axes than array has dimensions"
    )

    pad_width_list = [(pad_width, pad_width)] * arr.ndim
    for ax in range(arr.ndim):
        if ax not in padded_axes:
            pad_width_list[ax] = (0, 0)
    arr_pad = np.pad(arr, pad_width_list, mode=mode)
    return arr_pad


def _unpad_array(arr: np.array, pad_width: int, padded_axes: Sequence[int]):
    """Remove padding from the array."""

    assert len(padded_axes) <= arr.ndim, (
        "Cannot unpad more axes than array has dimensions"
    )

    unpad_slice = [
        slice(pad_width, arr.shape[axis] - pad_width)
        for axis, _ in enumerate(arr.shape)
    ]
    for ax in range(arr.ndim):
        if ax not in padded_axes:
            unpad_slice[ax] = slice(None)

    return arr[tuple(unpad_slice)]


def expand_attribution_channel(a_batch: np.ndarray, x_batch: np.ndarray):
    """Expand additional channel dimension(s) for attributions if needed."""
    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )
    if a_batch.ndim > x_batch.ndim:
        raise ValueError(f"a must not have greater ndim than x ({a_batch.ndim} > {x_batch.ndim})")

    if a_batch.ndim == x_batch.ndim:
        return a_batch
    else:
        attr_axes = infer_attribution_axes(a_batch, x_batch)

        #TODO: infer_attribution_axes currently returns dimensions w/o batch dimension
        attr_axes = [a+1 for a in attr_axes]
        expand_axes = [a for a in range(1, x_batch.ndim) if a not in attr_axes]

        return np.expand_dims(a_batch, axis=tuple(expand_axes))


def infer_attribution_axes(a_batch: np.ndarray, x_batch: np.ndarray) -> Sequence[int]:
    """
    Infers the axes in x_batch that are covered by a_batch.
    """
    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )

    if a_batch.ndim > x_batch.ndim:
        raise ValueError("Attributions need to have <= dimensions than inputs, but {} > {}".format(a_batch.ndim, x_batch.ndim))

    # TODO: we currently assume here that the batch axis is not carried into the perturbation functions
    # TODO: adapt to expanded attributions?
    a_shape = [s for s in np.shape(a_batch)[1:] if s != 1]
    x_shape = [s for s in np.shape(x_batch)[1:]]

    # Equal Shape
    if a_shape == x_shape:
        return np.arange(0, len(x_shape))

    # One attribution value per sample
    if len(a_shape) == 0:
        return np.array([])

    x_subshapes = [[x_shape[i] for i in range(start, start + len(a_shape))] for start in range(0, len(x_shape) - len(a_shape) + 1)]
    if x_subshapes.count(a_shape) < 1:
        # Check that attribution dimensions are (consecutive) subdimensions of inputs
        raise ValueError(
            "Attribution dimensions are not (consecutive) subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}".format(x_batch.shape, a_batch.shape)
        )
    elif x_subshapes.count(a_shape) > 1:
        # Check that attribution dimensions are (unique) subdimensions of inputs.
        # Consider potentially expanded dims in attributions.
        if a_batch.ndim == x_batch.ndim and len(a_shape) < a_batch.ndim:
            a_subshapes = [[np.shape(a_batch)[1:][i] for i in range(start, start + len(a_shape))] for start in
                           range(0, len(np.shape(a_batch)[1:]) - len(a_shape) + 1)]
            if a_subshapes.count(a_shape) == 1:
                # Inferring channel shape
                for dim in range(len(np.shape(a_batch)[1:]) + 1):
                    if a_shape == np.shape(a_batch)[1:][dim:]:
                        return np.arange(dim, len(np.shape(a_batch)[1:]))
                    if a_shape == np.shape(a_batch)[1:][:dim]:
                        return np.arange(0, dim)

            raise ValueError("Attribution axes could not be inferred for inputs of "
                             "shape {} and attributions of shape {}".format(x_batch.shape, a_batch.shape)
                             )

        raise ValueError(
            "Attribution dimensions are not unique subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}."
            "Please expand attribution dimensions for a unique solution".format(x_batch.shape, a_batch.shape)
        )
    else:
        # Inferring channel shape
        for dim in range(len(x_shape) + 1):
            if a_shape == x_shape[dim:]:
                return np.arange(dim, len(x_shape))
            if a_shape == x_shape[:dim]:
                return np.arange(0, dim)

    raise ValueError("Attribution axes could not be inferred for inputs of "
                     "shape {} and attributions of shape {}".format(x_batch.shape, a_batch.shape)
                     )


def expand_indices(arr: np.array, indices: Union[int, Sequence[int], Tuple[np.array]], indexed_axes: Sequence[int]) -> Tuple:
    """
    Expands indices to fit array shape. Returns expanded indices.
    """
    expanded_indices = np.array(indices) if not isinstance(indices, int) else np.array([indices])
    indexed_axes = np.sort(np.array(indexed_axes))

    asserts.assert_indexed_axes(arr, indexed_axes)

    # Check if unraveling is needed
    if indexed_axes.size != len(expanded_indices):
        if expanded_indices.ndim == 1:
            expanded_indices = np.unravel_index(expanded_indices, tuple([arr.shape[i] for i in indexed_axes]))
        else:
            raise ValueError("indices dimension doesn't match indexed_axes")

    # Handle case of 1D indices
    if not np.array(expanded_indices).ndim > 1:
        expanded_indices = [np.array(expanded_indices)]

    # Cast to list so item assignment works
    expanded_indices = list(expanded_indices)

    # Ensure array dimensions are kept when indexing.
    # Expands dimensions of each element in expanded_indices depending on the number of elements
    for i in range(len(expanded_indices)):
        expanded_indices[i] = np.expand_dims(expanded_indices[i], axis=tuple(range(len(expanded_indices)-1)))

    # Buffer with None-slices if indices index the last axes
    for i in range(0, indexed_axes[0]):
        expanded_indices = slice(None), *expanded_indices

    return tuple(expanded_indices)


def get_leftover_shape(arr: np.array, axes: Sequence[int]) -> Tuple:
    """
    Gets the shape of the arr dimensions not included in axes
    """
    axes = np.sort(np.array(axes))
    asserts.assert_indexed_axes(arr, axes)

    leftover_shape = tuple([arr.shape[i] for i in range(arr.ndim) if i not in axes])
    return leftover_shape