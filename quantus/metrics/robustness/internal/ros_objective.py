
import numpy as np
from quantus.functions.norm_func import lp_norm_2d, lp_norm_3d, lp_norm_4d


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

        num_dim = e_x.ndim
        if num_dim == 4:
            norm_function = lp_norm_4d
        elif num_dim == 3:
            norm_function = lp_norm_3d
        elif num_dim == 2:
            norm_function = lp_norm_2d
        else:
            raise ValueError(
                "Relative Output Stability only supports 4D, 3D and 2D inputs (batch dimension inclusive)."
            )

        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = norm_function(nominator)
        # fmt: on

        denominator = h_x - h_xs
        denominator = np.linalg.norm(denominator, axis=-1)
        denominator += (denominator == 0) * self._eps_min  # prevent division by 0
        return nominator / denominator