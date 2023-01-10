import pytest
from pytest_lazyfixture import lazy_fixture

from nlp.functions.perturbation_function import (
    spelling_replacement,
    synonym_replacement,
)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "text, method",
    [
        (lazy_fixture("sst2_dataset"), spelling_replacement),
        (lazy_fixture("ag_news_dataset"), spelling_replacement),
        (lazy_fixture("sst2_dataset"), synonym_replacement),
        (lazy_fixture("ag_news_dataset"), synonym_replacement),
    ],
)
def test_perturbation_function(text, method):
    result = method(text, k=1)  # noqa
    for i, j in zip(result, text):
        assert i != j
