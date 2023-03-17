# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import numpy as np


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    if scores.ndim > 2:
        raise ValueError("Only 2D and 1D inputs are supported.")
    scores = np.asarray(scores)
    scores = scores + np.finfo(np.float32).eps
    return (scores.T / np.abs(scores).sum(axis=-1)).T
