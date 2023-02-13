"""This modules holds a collection of perturbation functions i.e., ways to perturb an input or an explanation."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import time
import warnings

import numpy as np

from quantus.helpers.utils import get_name


def check_kwargs(kwargs):
    """
    Check that no additional kwargs are passed, i.e. the kwargs dict is empty.
    Raises an exception with helpful suggestions to fix the issue.

    Parameters
    ----------
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """
    if kwargs:
        raise ValueError(
            f"Please handle the following arguments: {kwargs}. "
            "There were unexpected keyword arguments passed to the metric method. "
            "Quantus has undergone heavy API-changes since the last release(s), "
            "to make the kwargs-passing and error handling more robust and transparent. "
            "Passing unexpected keyword arguments is now discouraged. Please adjust "
            "your code to pass your kwargs in dictionaries to the arguments named "
            "normalise_func_kwargs, explain_func_kwargs or model_predict_kwargs. "
            "For evaluate function pass explain_func_kwargs and call_kwargs."
            "And also, always make sure to check for typos. "
            "If these API changes are not suitable for your project's needs, "
            "please install quantus using 'pip install quantus==0.1.6' "
        )


def warn_noise_zero(noise: float) -> None:
    """
    Warn if noise is zero.

    Parameters
    ----------
    noise: float
        The amount of noise.

    Returns
    -------
    None
    """
    if noise == 0.0:
        print(
            f"Noise is set to {noise:.2f} which is likely to invalidate the evaluation outcome of the test"
            f" given that it depends on perturbation of input(s)/ attribution(s). "
            f"\n Recommended to re-parameterise the metric."
        )


def warn_absolute_operation(word: str = "") -> None:
    """
    Warn if an absolute operation is applied, where the metric is defined otherwise.

    Parameters
    ----------
    word: string
        A string for which is '' or 'not '.

    Returns
    -------
    None
    """
    print(
        f"An absolute operation should {word}be applied on the attributions, "
        "otherwise inconsistent results can be expected. Re-set 'abs' parameter."
    )


def warn_normalise_operation(word: str = "") -> None:
    """
    Warn if a normalisation operation is applied, where the metric is defined otherwise.

    Parameters
    ----------
    word: string
        A string for which is '' or 'not '.

    Returns
    -------
    None
    """
    print(
        f"A normalising operation should {word}be applied on the attributions, "
        "otherwise inconsistent results can be expected. Re-set 'normalise' parameter."
    )


def warn_segmentation(inside_attribution: float, total_attribution: float) -> None:
    """
    Warn if the inside explanation is greater than total explanation.

    Parameters
    ----------
    inside_attribution: float
        The size of inside attribution.
    total_attribution: float
        The size of total attribution.

    Returns
    -------
    None
    """
    warnings.warn(
        "Inside explanation is greater than total explanation"
        f" ({inside_attribution} > {total_attribution}), returning np.nan."
    )


def warn_empty_segmentation() -> None:
    """
    Warn if the segmentation mask is empty.

    Returns
    -------
    None
    """
    warnings.warn("Return np.nan as result as the segmentation map is empty.")


def warn_different_array_lengths() -> None:
    """
    Warn if the array lengths are different, for plotting.

    Returns
    -------
    None
    """
    warnings.warn(
        "The plotted measurements have different lengths. Clipping to minimum length."
    )


def warn_iterations_exceed_patch_number(n_iterations: int, n_patches: int) -> None:
    """
    Warn if the number of non-overlapping patches is lower than the number of iterations specified for this metric.

    Parameters
    ----------
    n_iterations: integer
        The number of iterations specified in the metric.
    n_patches: integer
        The number of patches specified in the metric.

    Returns
    -------
    None
    """
    if n_patches < n_iterations:
        warnings.warn(
            "The number of non-overlapping patches ({}) for this input and attribution"
            "is lower than the number of iterations specified for this metric ({})."
            "As a result, the number of measurements may vary for each input.".format(
                n_patches, n_iterations
            )
        )


def warn_parameterisation(
    metric_name: str = "Metric",
    sensitive_params: str = "X, Y and Z.",
    data_domain_applicability: str = "",
    citation: str = "INSERT CITATION",
):
    """
    Warn the parameterisation of the metric.

    Parameters
    ----------
    metric_name: string
        The metric name.
    sensitive_params: string
        The sensitive parameters of the metric.
    data_domain_applicability string
        The applicability when it comes to data domains, default = "".
    citation: string
        The citation.

    Returns
    -------
    None
    """

    time.sleep(1)

    print("Warnings and information:")
    text = (
        f" (1) The {get_name(metric_name)} metric is likely to be sensitive to the choice of "
        f"{sensitive_params}. {data_domain_applicability} \n (2) If attributions are normalised or their absolute values are taken it may "
        f"destroy or skew information in the explanation and as a result, affect the overall evaluation outcome."
        f"\n (3) Make sure to validate the choices for hyperparameters of the metric (by calling"
        f" .get_params of the metric instance).\n (4) For further information, see original publication: {citation}."
        f"\n (5) To disable these warnings set 'disable_warnings' = True when initialising the metric.\n"
    )
    print(text)


def deprecation_warnings(kwargs: dict) -> None:
    """
    Run deprecation warnings.

    Parameters
    ----------
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    None
    """

    text = "\n"
    if "img_size" in kwargs:
        text = (
            "argument '' is deprecated and has been removed from the current release.\n"
        )
    if "nr_channels" in kwargs:
        text = "argument 'max_steps_per_input' is deprecated and has been removed from the current release.\n"
    if "max_steps_per_input" in kwargs:
        text = "argument 'max_steps_per_input' is deprecated and has been removed from the current release.\n"
    if "pos_only" in kwargs:
        text = "argument 'pos_only' is deprecated and has been removed from the current release.\n"
    if "neg_only" in kwargs:
        text = "argument 'neg_only' is deprecated and has been removed from the current release.\n"

    if text != "\n":
        print(text)


def warn_perturbation_caused_no_change(x: np.ndarray, x_perturbed: np.ndarray) -> None:
    """
    Warn that perturbation applied to input caused change so that input and perturbed input is not the same.

    Parameters
    ----------
    x: np.ndarray
         The original input that is considered unperturbed.
    x_perturbed: np.ndarray
         The perturbed input.

    Returns
    -------
    None
    """
    if (x.flatten() != x_perturbed.flatten()).any():
        warnings.warn(
            "The settings for perturbing input e.g., 'perturb_func' "
            "didn't cause change in input. "
            "Reconsider the parameter settings."
        )


def warn_max_size() -> None:
    """
    Warns if the ratio is smaller than the maximum size, for attribution_localisaiton metric.
    Returns
    -------
    None
    """
    warnings.warn("Ratio is smaller than max size.")
