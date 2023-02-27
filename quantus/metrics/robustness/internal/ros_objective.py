
import numpy as np
from quantus.functions.norm_func import l2_norm


class RelativeOutputStabilityObjective:

    def __init__(self, eps_min: float):
        self._eps_min = eps_min

    def __call__(
        self,
        h_x: np.ndarray,
        h_xs: np.ndarray,
        e_x: np.ndarray,
        e_xs: np.ndarray,
    ) -> np.ndarray:
        """
        Computes relative output stabilities maximization objective
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

        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = l2_norm(nominator)
        # fmt: on

        denominator = h_x - h_xs
        denominator = l2_norm(denominator)
        denominator += (denominator == 0) * self._eps_min  # prevent division by 0
        return nominator / denominator