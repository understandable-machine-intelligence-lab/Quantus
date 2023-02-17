
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

        num_dim = e_x.ndim
        if num_dim == 4:
            norm_function = lp_norm_4d
        elif num_dim == 3:
            norm_function = lp_norm_3d
        elif num_dim == 2:
            norm_function = lp_norm_2d
        else:
            raise ValueError(
                "Relative Input Stability only supports 4D, 3D and 2D inputs (batch dimension inclusive)."
            )

        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = norm_function(nominator)
        # fmt: on
        denominator = l_x - l_xs
        denominator /= l_x + (l_x == 0) * self._eps_min  # prevent division by 0
        denominator = np.linalg.norm(denominator, axis=-1)
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator