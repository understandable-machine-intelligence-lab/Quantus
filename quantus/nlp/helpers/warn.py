from typing import List
import warnings


def warn_perturbation_caused_no_change(x: List[str], x_perturbed: List[str]) -> None:
    for index, (s1, s2) in enumerate(zip(x, x_perturbed)):
        if s1 == s2:
            warnings.warn(
                "The settings for perturbing input e.g., 'perturb_func' "
                f"didn't cause change in input at index {index}. "
                "Reconsider the parameter settings."
            )
