import pytest


from quantus.nlp.helpers.utils import choose_perturbation_function


@pytest.mark.nlp
def test_invaluid_perturbation_function():
    with pytest.raises(ValueError):
        choose_perturbation_function("dark_magic")
