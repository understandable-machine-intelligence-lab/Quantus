
import numpy as np
from quantus.functions.norm_func import lp_norm_2d, lp_norm_3d, lp_norm_4d


class RelativeRepresentationStabilityObjective:

    def __init__(self, eps_min: float):
        self._eps_min = eps_min

    def __call__(
            self,
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

        nominator_num_dim = e_x.ndim
        if nominator_num_dim == 4:
            nominator_norm_function = lp_norm_4d
        elif nominator_num_dim == 3:
            nominator_norm_function = lp_norm_3d
        elif nominator_num_dim == 2:
            nominator_norm_function = lp_norm_2d
        else:
            raise ValueError(
                "Relative Input Stability only supports 4D, 3D and 2D inputs (batch dimension inclusive)."
            )

        denominator_num_dim = l_x.ndim
        if denominator_num_dim == 4:
            denominator_norm_function = lp_norm_4d
        elif denominator_num_dim == 3:
            denominator_norm_function = lp_norm_3d
        elif denominator_num_dim == 2:
            denominator_norm_function = lp_norm_2d
        else:
            raise ValueError(
                "Relative Input Stability only supports 4D, 3D and 2D inputs (batch dimension inclusive)."
            )

        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = nominator_norm_function(nominator)
        # fmt: on
        denominator = l_x - l_xs
        denominator /= l_x + (l_x == 0) * self._eps_min  # prevent division by 0
        denominator = denominator_norm_function(denominator)
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator