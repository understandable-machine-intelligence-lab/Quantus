import numpy as np
from typing import Union, Callable


def attributes_check(metric):
    # https://towardsdatascience.com/5-ways-to-control-attributes-in-python-an-example-led-guide-2f5c9b8b1fb0
    attr = metric.__dict__
    if "perturb_func" in attr:
        if not callable(attr["perturb_func"]):
            raise TypeError("The 'perturb_func' must be a callable.")
    if "similarity_func" in attr:
        assert callable(attr["similarity_func"]), "The 'similarity_func' must be a callable."
    if "explain_func" in attr:
        assert callable(attr["explain_func"]), "The 'explain_func' must be a callable."
    if "normalize_func" in attr:
        assert callable(attr["normalize_func"]), "The 'normalize_func' must be a callable."
    if "text_warning" in attr:
        assert isinstance(attr["text_warning"], str), "The 'text_warning' function must be a string."
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
    y_pred: float, y_pred_perturb: float,
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


def assert_layer_order(layer_order: str) -> None:
    assert layer_order in ["top_down", "bottom_up", "independent"]


def assert_targets(x_batch: np.array,
                   y_batch: Union[np.array, int],) -> None:
    if not isinstance(y_batch, int):
        assert (np.shape(x_batch)[0] == np.shape(y_batch)[0]), "The 'y_batch' should by an integer or a list with " \
                                                               "the same number of samples as the 'x_batch' input."


def assert_attributions(x_batch: np.array,
                        a_batch: np.array) -> None:
    """Asserts on attributions."""
    assert type(a_batch) == np.ndarray, "Attributions 'a_batch' should be of type np.ndarray."
    assert (np.shape(x_batch)[0] == np.shape(a_batch)[0]), "The inputs 'x_batch' and attributions 'a_batch' should include the same number of samples."
    assert (np.shape(x_batch)[-1] == np.shape(a_batch)[-1]), "The inputs 'x_batch' and attributions 'a_batch' should share the same dimensions."


def assert_segmentations(x_batch: np.array,
                         s_batch: np.array) -> None:
    """Asserts on segmentations."""
    assert type(s_batch) == np.ndarray, "Segmentations 's_batch' should be of type np.ndarray."
    assert (np.shape(x_batch)[0] == np.shape(s_batch)[0]), "The inputs 'x_batch' and segmentations 's_batch' should include the same number of samples."
    assert (np.shape(x_batch)[-1] == np.shape(s_batch)[-1]), "The inputs 'x_batch' and segmentations 's_batch' should share the same dimensions."


def assert_max_size(max_size: float) -> None:
    assert ((max_size > 0.) and (max_size <= 1.)), "Set 'max_size' must be between 0. and 1."


def assert_plot_func(plot_func: Callable) -> None:
    assert callable(plot_func), "Make sure that 'plot_func' is a callable."


def assert_explain_func(explain_func: Callable) -> None:
    assert callable(explain_func), "Make sure 'explain_func' is a Callable that takes model, x_batch, " \
                                   "y_batch and **kwargs as arguments."