import numpy as np

from quantus.functions.norm_func import l2_norm


def relative_input_stability_objective(
    x: np.ndarray,
    xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min: float = 1e-6,
) -> np.ndarray:
    """
    Computes relative input stability's maximization objective
     as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.
    Parameters
    ----------
    x: np.ndarray
        Batch of images.
    xs: np.ndarray
        Batch of perturbed images.
    e_x: np.ndarray
        Explanations for x.
    e_xs: np.ndarray
        Explanations for xs.
    eps_min: float.
        Small constant to prevent division by 0 in relative_stability_objective, default 1e-6.

    Returns
    -------
    ris_obj: np.ndarray
        RIS maximization objective.
    """

    # fmt: off
    nominator = (e_x - e_xs) / (e_x + (e_x == 0) * eps_min)  # prevent division by 0
    nominator = l2_norm(nominator)
    # fmt: on

    denominator = x - xs
    denominator /= x + (x == 0) * eps_min
    # fmt: off
    denominator = l2_norm(denominator)
    # fmt: on
    denominator += (denominator == 0) * eps_min
    return nominator / denominator


def relative_output_stability_objective(
    h_x: np.ndarray,
    h_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min: float = 1e-6,
) -> np.ndarray:
    """
    Computes relative output stability's maximization objective
     as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.

    Parameters
    ----------
    h_x: np.ndarray
        Output logits for x_batch.
    h_xs: np.ndarray
        Output logits for xs_batch.
    e_x: np.ndarray
        Explanations for x.
    e_xs: np.ndarray
        Explanations for xs.
    eps_min: float.
        Small constant to prevent division by 0 in relative_stability_objective, default 1e-6.

    Returns
    -------
    ros_obj: np.ndarray
        ROS maximization objective.
    """

    # fmt: off
    nominator = (e_x - e_xs) / (e_x + (e_x == 0) * eps_min)  # prevent division by 0
    nominator = l2_norm(nominator)
    # fmt: on

    denominator = h_x - h_xs
    denominator = l2_norm(denominator)
    denominator += (denominator == 0) * eps_min  # prevent division by 0
    return nominator / denominator


def relative_representation_stability_objective(
    l_x: np.ndarray,
    l_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
    eps_min: float = 1e-6,
) -> np.ndarray:
    """
    Computes relative representation stabilities maximization objective
    as defined here https://arxiv.org/pdf/2203.06877.pdf by the authors.

    Parameters
    ----------
    l_x: np.ndarray
        Internal representation for x_batch.
    l_xs: np.ndarray
        Internal representation for xs_batch.
    e_x: np.ndarray
        Explanations for x.
    e_xs: np.ndarray
        Explanations for xs.
    eps_min: float.
        Small constant to prevent division by 0 in relative_stability_objective, default 1e-6.

    Returns
    -------
    rrs_obj: np.ndarray
        RRS maximization objective.
    """

    # fmt: off
    nominator = (e_x - e_xs) / (e_x + (e_x == 0) * eps_min)  # prevent division by 0
    nominator = l2_norm(nominator)
    # fmt: on
    denominator = l_x - l_xs
    denominator /= l_x + (l_x == 0) * eps_min  # prevent division by 0
    denominator = l2_norm(denominator)
    denominator += (denominator == 0) * eps_min
    return nominator / denominator
