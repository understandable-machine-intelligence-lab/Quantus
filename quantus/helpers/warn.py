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
            f"Unexpected keyword arguments encountered: {kwargs}. "
            "To ensure proper usage, please refer to the 'get_params' method of the initialised metric instance "
            "or consult the Quantus documentation. Avoid passing extraneous keyword arguments. "
            "Ensure that your metric arguments are correctly structured, particularly 'normalise_func_kwargs', "
            "'explain_func_kwargs', and 'model_predict_kwargs'. Additionally, always verify for any typos."
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
    Warn that perturbation applied to input caused no change so that input and perturbed input is the same.

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
    if np.allclose(x, x_perturbed, equal_nan=True):
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


def warn_attributions(x_batch: np.array, a_batch: np.array) -> None:
    """
    Asserts on attributions, assumes channel first layout.

    Parameters
    ----------
    x_batch: np.ndarray
         The batch of input to compare the shape of the attributions with.
    a_batch: np.ndarray
         The batch of attributions.

    Returns
    -------
    None
    """
    if not (type(a_batch) == np.ndarray):
        warnings.warn("Attributions 'a_batch' should be of type np.ndarray.")
    if np.shape(x_batch)[0] == np.shape(a_batch)[0]:
        warnings.warn(
            "The inputs 'x_batch' and attributions 'a_batch' should "
            "include the same number of samples."
            "{} != {}".format(np.shape(x_batch)[0], np.shape(a_batch)[0])
        )
    if not np.ndim(x_batch) == np.ndim(a_batch):
        warnings.warn(
            "The inputs 'x_batch' and attributions 'a_batch' should "
            "have the same number of dimensions."
            "{} != {}".format(np.ndim(x_batch), np.ndim(a_batch))
        )
    a_shape = [s for s in np.shape(a_batch)[1:] if s != 1]
    x_shape = [s for s in np.shape(x_batch)[1:]]
    if not (a_shape[0] == x_shape[0] or a_shape[-1] == x_shape[-1]):
        warnings.warn(
            "The dimensions of attribution and input per sample should correspond in either "
            "the first or last dimensions, but got shapes "
            "{} and {}".format(a_shape, x_shape)
        )
    if not all([a in x_shape for a in a_shape]):
        warnings.warn(
            "All attribution dimensions should be included in the input dimensions, "
            "but got shapes {} and {}".format(a_shape, x_shape)
        )
    if not all(
        [
            x_shape.index(a) > x_shape.index(a_shape[i])
            for a in a_shape
            for i in range(a_shape.index(a))
        ]
    ):
        warnings.warn(
            "The dimensions of the attribution must correspond to dimensions of the input in the same order, "
            "but got shapes {} and {}".format(a_shape, x_shape)
        )
    if np.all((a_batch == 0)):
        warnings.warn(
            "The elements in the attribution vector are all equal to zero, "
            "which may cause inconsistent results since many metrics rely on ordering. "
            "Recompute the explanations."
        )
    if np.all((a_batch == 1.0)):
        warnings.warn(
            "The elements in the attribution vector are all equal to one, "
            "which may cause inconsistent results since many metrics rely on ordering. "
            "Recompute the explanations."
        )
    if len(set(a_batch.flatten().tolist())) > 1:
        warnings.warn(
            "The attributions are uniformly distributed, "
            "which may cause inconsistent results since many "
            "metrics rely on ordering."
            "Recompute the explanations."
        )
    if np.all((a_batch < 0.0)):
        warnings.warn("Attributions should not all be less than zero.")
