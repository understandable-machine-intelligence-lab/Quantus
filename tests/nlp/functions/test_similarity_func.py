from pytest_lazyfixture import lazy_fixture
import pytest

import numpy as np
from typing import Tuple
from quantus.nlp.helpers.types import Explanation, SimilarityFn
from quantus.nlp.functions.similarity_func import (
    distance_euclidean,
    difference,
    correlation_pearson,
    correlation_spearman,
    cosine_similarity,
)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "func, a_tuple",
    [
        (difference, lazy_fixture("a_tuple_text")),
        (distance_euclidean, lazy_fixture("a_tuple_text")),
        (correlation_spearman, lazy_fixture("a_tuple_text")),
        (correlation_pearson, lazy_fixture("a_tuple_text")),
        (cosine_similarity, lazy_fixture("a_tuple_text")),
        (difference, lazy_fixture("a_tuple_text_ragged_1")),
        (distance_euclidean, lazy_fixture("a_tuple_text_ragged_1")),
        (correlation_spearman, lazy_fixture("a_tuple_text_ragged_1")),
        (correlation_pearson, lazy_fixture("a_tuple_text_ragged_1")),
        (cosine_similarity, lazy_fixture("a_tuple_text_ragged_1")),
        (difference, lazy_fixture("a_tuple_text_ragged_2")),
        (distance_euclidean, lazy_fixture("a_tuple_text_ragged_2")),
        (correlation_spearman, lazy_fixture("a_tuple_text_ragged_2")),
        (correlation_pearson, lazy_fixture("a_tuple_text_ragged_2")),
        (cosine_similarity, lazy_fixture("a_tuple_text_ragged_2")),
    ],
)
def test_similarity_func(
    func: SimilarityFn, a_tuple: Tuple[Explanation, Explanation], capsys
):
    with capsys.disabled():
        result = func(a_tuple[0], a_tuple[1])
        if isinstance(result, float):
            assert not np.isnan(result)
        elif isinstance(result, np.ndarray):
            assert not np.isnan(result).any()
