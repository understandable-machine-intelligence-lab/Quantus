import numpy as np
from typing import Union, Callable


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


def set_warn(call):
    # TODO. Implement warning logic of decorator if text_warning is an attribute in class.
    def call_fn(*args):
        return call_fn

    return call
    # attr = call.__dict__
    # print(dir(call))
    # attr = {}
    # if "text_warning" in attr:
    #    call.print_warning(text=attr["text_warning"])
    # else:
    #    print("Do nothing.")
    #    pass


def assert_features_in_step(features_in_step: int, img_size: int) -> None:
    """Assert that features in step is compatible with the image size."""
    assert (img_size * img_size) % features_in_step == 0, (
        "Set 'features_in_step' so that the modulo remainder "
        "returns zero given the img_size."
    )


def assert_max_steps(max_steps_per_input: int, img_size: int) -> None:
    """Assert that max steps per inputs is compatible with the image size."""
    assert (img_size * img_size) % max_steps_per_input == 0, (
        "Set 'max_steps_per_input' so that the modulo remainder "
        "returns zero given the img_size."
    )


def assert_patch_size(patch_size: int, img_size: int) -> None:
    """Assert that patch size that are not compatible with input size."""
    assert (
        img_size % patch_size == 0
    ), "Set 'patch_size' so that the modulo remainder returns 0 given the image size."


def assert_attributions_order(order: str) -> None:
    """Assert that order is in pre-defined list."""
    assert order in [
        "morf",
        "lorf",
    ], "The order of sorting the attributions must be either morf or lorf-"


def assert_nr_segments(nr_segments: int) -> None:
    """Assert that the number of segments given the segmentation algorithm is more than one."""
    assert (
        nr_segments > 1
    ), "The number of segments from the segmentation algorithm must be more than one."


def assert_perturbation_caused_change(x: np.ndarray, x_perturbed: np.ndarray) -> None:
    """Assert that perturbation applied to input caused change so that input and perturbed input is not the smae."""
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
    """Asserts on attributions."""
    assert (
        type(a_batch) == np.ndarray
    ), "Attributions 'a_batch' should be of type np.ndarray."
    assert np.shape(x_batch)[0] == np.shape(a_batch)[0], (
        "The inputs 'x_batch' and attributions 'a_batch' should "
        "include the same number of samples."
        "{} != {}".format(np.shape(x_batch)[0], np.shape(a_batch)[0])
    )

    assert np.shape(x_batch)[-2:] == np.shape(a_batch)[-2:], (
        "The inputs 'x_batch' and attributions 'a_batch' "
        "should share the same dimensions."
        "{} != {}".format(np.shape(x_batch)[-1], np.shape(a_batch)[-1])
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


def assert_segmentations(x_batch: np.array, s_batch: np.array) -> None:
    """Asserts on segmentations."""
    assert (
        type(s_batch) == np.ndarray
    ), "Segmentations 's_batch' should be of type np.ndarray."
    assert (
        np.shape(x_batch)[0] == np.shape(s_batch)[0]
    ), "The inputs 'x_batch' and segmentations 's_batch' should include the same number of samples."
    assert (
        np.shape(x_batch)[-2:] == np.shape(s_batch)[-2:]
    ), "The inputs 'x_batch' and segmentations 's_batch' should share the same dimensions."
    assert (
        np.shape(s_batch)[1] == 1
    ), "The second dimension of the segmentations 's_batch' should be equal to 1."
    # TODO.
    assert (
        len(np.nonzero(s_batch)) > 0
    ), "The segmentation 's_batch' must contain non-zero elements."
    assert (
        np.isin(s_batch.flatten(), [0, 1]).all()
        or np.isin(s_batch.flatten(), [True, False]).all()
    ), ("The " "segmentation 's_batch' should not only contain [1‚0] or [True, False].")


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
