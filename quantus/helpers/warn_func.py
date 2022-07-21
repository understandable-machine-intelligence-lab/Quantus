"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import time
import warnings
from termcolor import colored
from .utils import get_name


def warn_noise_zero(noise: float) -> None:
    if noise == 0.0:
        print(
            f"Noise is set to {noise:.2f} which is likely to invalidate the evaluation outcome of the test"
            f" given that it depends on perturbation of input(s)/ attribution(s). "
            f"\n Recommended to re-parameterise the metric."
        )


def warn_absolutes_applied() -> None:
    print(
        "An absolute operation is applied on the attributions (regardless of set 'abs' parameter) "
        "since otherwise inconsistent results can be expected."
    )


def warn_absolutes_requirement() -> None:
    print(
        "An absolute operation is applied on the attributions (regardless of set 'abs' parameter) "
        "since it is required by the metric."
    )


def warn_absolutes_skipped() -> None:
    print(
        "An absolute operation on the attributions is skipped "
        "since inconsistent results can be expected if applied."
    )


def warn_normalisation_skipped() -> None:
    print(
        "A normalising operation on the attributions is skipped "
        "since inconsistent results can be expected if applied."
    )


def warn_parameterisation(
    metric_name: str = "Metric",
    sensitive_params: str = "X, Y and Z.",
    citation: str = "INSERT CITATION",
):

    time.sleep(1)

    print("Warnings and information:")
    text = (
        f" (1) The {get_name(metric_name)} metric is likely to be sensitive to the choice of "
        f"{sensitive_params}. \n (2) If attributions are normalised or their absolute values are taken it may "
        f"destroy or skew information in the explanation and as a result, affect the overall evaluation outcome."
        f"\n (3) Make sure to validate the choices for hyperparameters of the metric (by calling"
        f" .get_params of the metric instance).\n (4) For further information, see original publication: {citation}."
        f"\n (5) To disable these warnings set 'disable_warnings' = True when initialising the metric.\n"
    )
    print(colored(text=text))


def deprecation_warnings(kwargs: dict = {}) -> None:
    text = "\n"
    if "img_size" in kwargs:
        text += "argument 'img_size' is deprecated and will be removed in future versions.\n"
    if "nr_channels" in kwargs:
        text = "argument 'nr_channels' is deprecated and will be removed in future versions.\n"
    if "max_steps_per_input" in kwargs:
        text = "argument 'max_steps_per_input' is deprecated and will be removed in future versions.\n"

    if text != "\n":
        print(text)
