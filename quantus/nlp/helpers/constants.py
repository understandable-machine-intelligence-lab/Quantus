from typing import List, Dict
from quantus.nlp.metrics.robustness.avg_sensitivity import AvgSensitivity
from quantus.nlp.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.nlp.metrics.robustness.relative_input_stability import (
    RelativeInputStability,
)
from quantus.nlp.metrics.robustness.relative_output_stability import (
    RelativeOutputStability,
)
from quantus.nlp.metrics.robustness.relative_representation_stability import (
    RelativeRepresentationStability,
)
from quantus.nlp.metrics.randomisation.model_parameter_randomisation import (
    ModelParameterRandomisation,
)
from quantus.nlp.metrics.randomisation.random_logit import RandomLogit
from quantus.nlp.metrics.faithfullness.token_flipping import TokenFlipping

from quantus.nlp.functions.perturb_func import (
    spelling_replacement,
    synonym_replacement,
    typo_replacement,
    uniform_noise,
    gaussian_noise,
)
from quantus.nlp.functions.normalise_func import normalize_sum_to_1

AVAILABLE_METRICS = {
    "Robustness": {
        "Max-Sensitivity": MaxSensitivity,
        "Avg-Sensitivity": AvgSensitivity,
        "Relative Input Stability": RelativeInputStability,
        "Relative Output Stability": RelativeOutputStability,
        "Relative Representation Stability": RelativeRepresentationStability,
    },
    "Randomisation": {
        "Model Parameter Randomisation": ModelParameterRandomisation,
        "Random Logit": RandomLogit,
    },
    "Faithfulness": {"Token Flipping": TokenFlipping},
}

METRICS_SUPPORT_PLAIN_TEXT_PERTURBATION = {
    "Max-Sensitivity": MaxSensitivity,
    "Avg-Sensitivity": AvgSensitivity,
    "Relative Input Stability": RelativeInputStability,
    "Relative Output Stability": RelativeOutputStability,
    "Relative Representation Stability": RelativeRepresentationStability,
}

METRICS_SUPPORT_NUMERICAL_PERTURBATION = {
    "Max-Sensitivity": MaxSensitivity,
    "Avg-Sensitivity": AvgSensitivity,
    "Relative Input Stability": RelativeInputStability,
    "Relative Output Stability": RelativeOutputStability,
    "Relative Representation Stability": RelativeRepresentationStability,
}

AVAILABLE_PLAIN_TEXT_PERTURBATION_FUNCTIONS = {
    "spelling_replacement": spelling_replacement,
    "synonym_replacement": synonym_replacement,
    "typo_replacement": typo_replacement,
}

AVAILABLE_LATENT_SPACE_PERTURBATION_FUNCTIONS = {
    "uniform_noise": uniform_noise,
    "gaussian_noise": gaussian_noise,
}

AVAILABLE_PERTURBATION_FUNCTIONS = {
    **AVAILABLE_PLAIN_TEXT_PERTURBATION_FUNCTIONS,
    **AVAILABLE_LATENT_SPACE_PERTURBATION_FUNCTIONS,
}

AVAILABLE_NORMALISATION_FUNCTIONS = {"normalize_sum_to_1": normalize_sum_to_1}

AVAILABLE_PLAIN_TEXT_XAI_METHODS = [
    "GradNorm",
    "GradXInput",
    "IntGrad",
    "NoiseGrad++",
    "LIME",
    "SHAP",
]

AVAILABLE_NUMERICAL_XAI_METHODS = [
    "GradNorm",
    "GradXInput",
    "IntGrad",
    "NoiseGrad++",
]

AVAILABLE_XAI_METHODS = (
    AVAILABLE_PLAIN_TEXT_XAI_METHODS + AVAILABLE_NUMERICAL_XAI_METHODS
)


def available_categories() -> List[str]:
    """
    Retrieve the available metric categories in Quantus.

    Returns
    -------
    List[str]
        With the available metric categories in Quantus.
    """
    return [c for c in AVAILABLE_METRICS.keys()]  # pragma: not covered


def available_metrics() -> Dict[str, List[str]]:
    """
    Retrieve the available metrics in Quantus.

    Returns
    -------
    Dict[str, str]
        With the available metrics, under each category in Quantus.
    """
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}  # type: ignore  # pragma: not covered


def available_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS]  # pragma: not covered


def available_plain_text_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods, which can be applied to plain text inputs.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_PLAIN_TEXT_XAI_METHODS]  # pragma: not covered


def available_numerical_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods, which can be applied to numerical representations of inputs.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_NUMERICAL_XAI_METHODS]  # pragma: not covered


def available_perturbation_functions() -> List[str]:
    """
    Retrieve the available perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]  # pragma: not covered


def available_plain_text_perturbation_functions() -> List[str]:
    return [
        c for c in AVAILABLE_PLAIN_TEXT_PERTURBATION_FUNCTIONS.keys()
    ]  # pragma: not covered


def available_latent_space_perturbation_functions() -> List[str]:
    return [c for c in AVAILABLE_LATENT_SPACE_PERTURBATION_FUNCTIONS.keys()]


def available_normalisation_functions() -> List[str]:
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]  # pragma: not covered


def available_metrics_plain_text_perturbation() -> List[str]:
    return [k for k in METRICS_SUPPORT_PLAIN_TEXT_PERTURBATION.keys()]


def available_metrics_numerical_perturbation() -> List[str]:
    return [k for k in METRICS_SUPPORT_NUMERICAL_PERTURBATION]
