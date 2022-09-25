import numpy as np
from typing import Callable, Tuple, Union, Sequence


def attributes_check(metric):
    # https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0
    attr = metric.__dict__
    if "perturb_func" in attr:
        if not callable(attr["perturb_func"]):
            raise TypeError("The 'perturb_func' must be a callable.")
    if "similarity_func" in attr:
        assert callable(
            attr["similarity_func"]
        ), "The 'similarity_func' must be a callable."
    if "explain_func" in attr:
        assert callable(attr["explain_func"]), "The 'explain_func' must be a callable."
    if "normalize_func" in attr:
        assert callable(
            attr["normalize_func"]
        ), "The 'normalize_func' must be a callable."
    if "text_warning" in attr:
        assert isinstance(
            attr["text_warning"], str
        ), "The 'text_warning' function must be a string."
    return metric


def assert_model_predictions_deviations(
    y_pred: float, y_pred_perturb: float, threshold: float = 0.01
):
    """Check that model predictions does not deviate more than a given threshold."""
    if abs(y_pred - y_pred_perturb) > threshold:
        return True
    else:
        return False


def assert_model_predictions_correct(
    y_pred: float,
    y_pred_perturb: float,
):
    """Assert that model predictions are the same."""
    if y_pred == y_pred_perturb:
        return True
    else:
        return False


def assert_features_in_step(
    features_in_step: int, input_shape: Tuple[int, ...]
) -> None:
    """Assert that features in step is compatible with the image size."""
    assert np.prod(input_shape) % features_in_step == 0, (
        "Set 'features_in_step' so that the modulo remainder "
        "returns zero given the product of the input shape."
        f" ({np.prod(input_shape)} % {features_in_step} != 0)"
    )


def assert_patch_size(patch_size: int, shape: Tuple[int, ...]) -> None:
    """Assert that patch size is compatible with given shape."""
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
    if np.prod(shape) % np.prod(patch_size) != 0:
        raise ValueError(
            "Set 'patch_size' so that the input shape modulo remainder returns 0"
            f" [np.prod({shape}) % np.prod({patch_size}) != 0"
            f" => {np.prod(shape)} % {np.prod(patch_size)} != 0]"
        )


def assert_attributions_order(order: str) -> None:
    """Assert that order is in pre-defined list."""
    assert order in [
        "random",
        "morf",
        "lorf",
    ], "The order of sorting the attributions must be either random, morf, or lorf."


def assert_nr_segments(nr_segments: int) -> None:
    """Assert that the number of segments given the segmentation algorithm is more than one."""
    assert (
        nr_segments > 1
    ), "The number of segments from the segmentation algorithm must be more than one."


def assert_perturbation_caused_change(x: np.ndarray, x_perturbed: np.ndarray) -> None:
    """Assert that perturbation applied to input caused change so that input and perturbed input is not the same."""
    assert (x.flatten() != x_perturbed.flatten()).any(), (
        "The settings for perturbing input e.g., 'perturb_func' "
        "didn't cause change in input. "
        "Reconsider the parameter settings."
    )


def assert_layer_order(layer_order: str) -> None:
    """Assert that layer order is in pre-defined list."""
    assert layer_order in ["top_down", "bottom_up", "independent"]


def assert_targets(
    x_batch: np.array,
    y_batch: np.array,
) -> None:
    if not isinstance(y_batch, int):
        assert np.shape(x_batch)[0] == np.shape(y_batch)[0], (
            "The 'y_batch' should by an integer or a list with "
            "the same number of samples as the 'x_batch' input"
            "{} != {}".format(np.shape(x_batch)[0], np.shape(y_batch)[0])
        )


def assert_attributions(x_batch: np.array, a_batch: np.array) -> None:
    """Asserts on attributions. Assumes channel first layout."""
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
    """Asserts on segmentations."""
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


def assert_max_size(max_size: float) -> None:
    assert (max_size > 0.0) and (
        max_size <= 1.0
    ), "Set 'max_size' must be between 0. and 1."


def assert_plot_func(plot_func: Callable) -> None:
    assert callable(plot_func), "Make sure that 'plot_func' is a callable."


def assert_explain_func(explain_func: Callable) -> None:
    assert callable(explain_func), (
        "Make sure 'explain_func' is a Callable that takes model, x_batch, "
        "y_batch and **kwargs as arguments."
    )


def assert_value_smaller_than_input_size(x: np.ndarray, value: int, value_name: str):
    """Checks if value is smaller than input size.
    Assumes batch and channel first dimension
    """
    if value >= np.prod(x.shape[2:]):
        raise ValueError(
            f"'{value_name}' must be smaller than input size."
            f" [{value} >= {np.prod(x.shape[2:])}]"
        )


# TODO: Change for batching update, since currently single images are expected.
def assert_indexed_axes(arr: np.array, indexed_axes: Sequence[int]):
    """
    Checks that indexed_axes fits arr
    """
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
