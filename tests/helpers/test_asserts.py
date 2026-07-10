import numpy as np
import pytest

from quantus.helpers.asserts import assert_attributions


@pytest.fixture
def x_batch():
    return np.random.uniform(0, 1, size=(2, 1, 4, 4))


@pytest.fixture
def all_negative_a_batch():
    return -np.random.uniform(0, 1, size=(2, 1, 4, 4))


@pytest.mark.asserts
def test_assert_attributions_rejects_all_negative_by_default(
    x_batch, all_negative_a_batch
):
    with pytest.raises(AssertionError, match="should not all be less than zero"):
        assert_attributions(x_batch=x_batch, a_batch=all_negative_a_batch)


@pytest.mark.asserts
def test_assert_attributions_allows_all_negative_when_check_disabled(
    x_batch, all_negative_a_batch
):
    # Should not raise: metrics such as Infidelity use attributions in a signed
    # dot product and set Metric.allow_negative_attributions = True to opt out
    # of this specific check.
    assert_attributions(
        x_batch=x_batch, a_batch=all_negative_a_batch, check_all_negative=False
    )
