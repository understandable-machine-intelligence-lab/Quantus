import numpy as np
from quantus.functions.norm_func import l2_norm


class RelativeInputStabilityObjective:

    def __init__(self, eps_min: float):
        self._eps_min = eps_min

    def __call__(
            self, x: np.ndarray, xs: np.ndarray, e_x: np.ndarray, e_xs: np.ndarray
    ) -> np.ndarray:
        """
        Computes relative input stabilities maximization objective
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

        # fmt: off
        nominator = (e_x - e_xs) / (e_x + (e_x == 0) * self._eps_min)  # prevent division by 0
        nominator = l2_norm(nominator)
        # fmt: on

        denominator = x - xs
        denominator /= x + (x == 0) * self._eps_min
        # fmt: off
        denominator = l2_norm(denominator)
        # fmt: on
        denominator += (denominator == 0) * self._eps_min
        return nominator / denominator


