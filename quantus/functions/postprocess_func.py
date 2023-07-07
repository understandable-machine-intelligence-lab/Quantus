"""This module provides some functionality to postprocess different evaluation outcomes."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import List, Callable
import warnings
import numpy as np

from quantus.helpers.asserts import assert_skill_scores
from quantus.helpers.enums import ScoreDirection


def explanation_skill_score(
    y_scores: np.array,
    y_refs: np.array,
    score_direction: ScoreDirection = ScoreDirection.HIGHER,
    agg_func: Callable = lambda x: x,
) -> List[float]:
    """
    Calculate the Explanation Skill Score (ESS).

    Parameters
    ----------
    y_scores: np.array
        Actual score outcome.
    y_refs: np.array
        Reference score, worst-case scenario.
    score_direction: ScoreDirection
        The direction of evaluation scores (i.e., lower or higher are considered better).
    agg_func: callable
        Indicates how to aggregates scores e.g., pass np.mean.

    Returns
    -------
    score: list of floats
        Explanation skill score
    """

    # Make asserts.
    assert_skill_scores(y_scores=y_scores, y_refs=y_refs)

    # Retrieve the optimal value.
    if score_direction == ScoreDirection.HIGHER:
        optimal_value = 1.0

        # Verify that all elements are not larger than one.
        assert np.all(y_scores <= 1), "For skill score calculation, 'y_scores' cannot contain values larger than 1."
        assert np.all(y_refs <= 1), "For skill score calculation, 'y_refs' cannot contain values larger than 1."

    elif score_direction == ScoreDirection.LOWER:
        optimal_value = 0.0

    else:
        raise ValueError(
            "To calculate skill score, the 'score_direction' must be either "
            "'ScoreDirection.HIGHER' or 'ScoreDirection.LOWER'"
        )

    skill_scores = []
    for i in range(len(y_scores)):

        try:
            ss = (y_scores[i] - y_refs[i]) / (optimal_value - y_refs[i])
        except RuntimeWarning:
            ss = 0.0

        skill_scores.append(ss)

    return agg_func(skill_scores)