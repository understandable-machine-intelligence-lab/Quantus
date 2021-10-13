"""This modules holds a collection of perturbation functions i..e, ways to perturb an input or an explanation."""
import time
from termcolor import colored


def warn_attributions(normalise: bool, abs: bool) -> None:
    if not normalise and not abs:
        pass
    else:
        text = ""
        if normalise:
            text += "Normalising attributions"
            if abs:
                text += " and taking their absolute values"
        if not normalise and abs:
            text += "Taking the absolute values of attributions"
        text += " may destroy or skew information in the explanation and as a result, affect the overall evaluation " \
                "outcome.\n"
        print(text)


def warn_noise_zero(noise: float) -> None:
    if noise == 0.0:
        print(
            f"Noise is set to {noise:.2f} which is likely to invalidate the evaluation outcome of the test"
            f" given that it depends on perturbation of input(s)/ attribution(s). "
            f"\n Recommended to re-parameterise the metric."
        )

def warn_parameterisation(text: str):
    time.sleep(2)
    print("WARNINGS.")
    print(colored(text=text)) #warnings.warn(colored(text=text, color="blue"), category=Warning)
