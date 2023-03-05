import numpy as np

from quantus.functions.norm_func import l2_norm


def relative_input_stability_objective(
    x: np.ndarray,
    xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
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

    Returns
    -------
    ris_obj: np.ndarray
        RIS maximization objective.
    """

    # prevent division by 0
    eps_min = np.finfo(np.float32).eps

    nominator = (e_x - e_xs) / (e_x + eps_min)
    nominator = l2_norm(nominator)

    denominator = x - xs
    denominator /= x + eps_min
    denominator = l2_norm(denominator) + eps_min

    return nominator / denominator


def relative_output_stability_objective(
    h_x: np.ndarray,
    h_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
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

    Returns
    -------
    ros_obj: np.ndarray
        ROS maximization objective.
    """

    # prevent division by 0
    eps_min = np.finfo(np.float32).eps

    nominator = (e_x - e_xs) / (e_x + eps_min)
    nominator = l2_norm(nominator)

    denominator = h_x - h_xs
    denominator = l2_norm(denominator) + eps_min
    return nominator / denominator


def relative_representation_stability_objective(
    l_x: np.ndarray,
    l_xs: np.ndarray,
    e_x: np.ndarray,
    e_xs: np.ndarray,
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

    Returns
    -------
    rrs_obj: np.ndarray
        RRS maximization objective.
    """

    # prevent division by 0
    eps_min = np.finfo(np.float32).eps

    nominator = (e_x - e_xs) / (e_x + eps_min)
    nominator = l2_norm(nominator)

    denominator = l_x - l_xs
    denominator /= (l_x + eps_min)
    denominator = l2_norm(denominator) + eps_min

    return nominator / denominator