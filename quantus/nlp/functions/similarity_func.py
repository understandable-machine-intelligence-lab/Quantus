import numpy as np

from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import pad_ragged_vector
import quantus.functions.similarity_func as similarity_func


def difference(a: Explanation, b: Explanation, padded: bool = True) -> np.ndarray:
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    return a_padded - b_padded


def distance_euclidean(a: Explanation, b: Explanation, padded: bool = True) -> float:
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    return similarity_func.distance_euclidean(a_padded, b_padded)


def cosine_similarity(a: Explanation, b: Explanation, padded: bool = True) -> float:
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    return similarity_func.cosine(a_padded, b_padded)


def correlation_spearman(a: Explanation, b: Explanation, padded: bool = True) -> float:
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    return similarity_func.correlation_spearman(a_padded, b_padded)


def correlation_pearson(a: Explanation, b: Explanation, padded: bool = True) -> float:
    if not padded:
        raise NotImplementedError()
    a_padded, b_padded = pad_ragged_vector(a[1], b[1])
    return similarity_func.correlation_pearson(a_padded, b_padded)
