"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import time
from termcolor import colored
from .utils import get_name


def warn_attributions(normalise: bool, abs: bool) -> None:
    if not normalise and not abs:
        pass
    else:
        text = "\n"
        if normalise:
            text += "Normalising attributions"
            if abs:
                text += " and taking their absolute values"
        if not normalise and abs:
            text += "Taking the absolute values of attributions"
        text += (
            " may destroy or skew information in the explanation and as a result, affect the overall evaluation "
            "outcome.\n"
        )
        print(text)


def warn_noise_zero(noise: float) -> None:
    if noise == 0.0:
        print(
            f"Noise is set to {noise:.2f} which is likely to invalidate the evaluation outcome of the test"
            f" given that it depends on perturbation of input(s)/ attribution(s). "
            f"\n Recommended to re-parameterise the metric."
        )


def warn_parameterisation(
    metric_name: str = "Metric",
    sensitive_params: str = "X, Y and Z.",
    citation: str = "INSERT CITATION",
):
    time.sleep(2)
    print("WARNINGS.")
    text = (
        f"\nThe {get_name(metric_name)} metric is likely to be sensitive to the choice of "
        f"{sensitive_params}. \nTo avoid misinterpretation of scores, consider all relevant hyperparameters of "
        f"the metric (by calling .get_params of the metric instance). \nFor further reading see: {citation}."
        f"\nTo disable warnings set 'disable_warnings'=True in the initialisation of the metric instance.\n"
    )
    print(
        colored(text=text)
    )  # warnings.warn(colored(text=text, color="blue"), category=Warning)
