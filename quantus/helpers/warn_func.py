"""This modules holds a collection of perturbation functions i.e., ways to perturb an input or an explanation."""
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


def warn_absolute_operation(word: str = "") -> None:
    print(
        f"An absolute operation should {word}be applied on the attributions, "
        "otherwise inconsistent results can be expected! Re-set 'abs' parameter accordingly."
    )

def warn_normalise_operation(word: str = "") -> None:
    print(
        f"A normalising operation should {word}be applied on the attributions, "
        "otherwise inconsistent results can be expected! Re-set 'normalise' parameter accordingly."
    )


def warn_segmentation(inside_attribution, total_attribution) -> None:
    warnings.warn(
        "Inside explanation is greater than total explanation"
        f" ({inside_attribution} > {total_attribution}), returning np.nan."
    )

def warn_empty_segmentation() -> None:
    warnings.warn(
        "Return np.nan as result as the segmentation map is empty."
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
