from typing import List, NamedTuple
import numpy as np


class TokenSalience(NamedTuple):
    tokens: List[str]
    salience: np.ndarray
