"""This module contains constants and simple methods to retreive the available metrics, perturbation-,
similarity-, normalisation- functions and explanation methods in Quantus."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import List, Dict
from quantus.nlp.metrics.robustness.avg_sensitivity import AvgSensitivity
from quantus.nlp.metrics.robustness.max_sensitivity import MaxSensitivity
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
    },
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
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> Dict[str, List[str]]:
    """
    Retrieve the available metrics in Quantus.

    Returns
    -------
    Dict[str, str]
        With the available metrics, under each category in Quantus.
    """
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS]


def available_plain_text_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods, which can be applied to plain text inputs.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_PLAIN_TEXT_XAI_METHODS]


def available_numerical_xai_methods() -> List[str]:
    """
    Retrieve the available explanation methods, which can be applied to numerical representations of inputs.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_NUMERICAL_XAI_METHODS]


def available_perturbation_functions() -> List[str]:
    """
    Retrieve the available perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_plain_text_perturbation_functions() -> List[str]:
    """
    Retrieve the available plain-text perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_PLAIN_TEXT_PERTURBATION_FUNCTIONS.keys()]


def available_latent_space_perturbation_functions() -> List[str]:
    """
    Retrieve the available perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_LATENT_SPACE_PERTURBATION_FUNCTIONS.keys()]


def available_normalisation_functions() -> List[str]:
    """
    Retrieve the available normalisation functions in Quantus.

    Returns
    -------
    List[str]
        With the available normalisation functions in Quantus.
    """
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]
