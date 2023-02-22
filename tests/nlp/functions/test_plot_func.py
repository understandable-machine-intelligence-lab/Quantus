import pytest
from quantus.nlp.functions.plot_func import (
    visualise_explanations_as_html,
    visualise_explanations_as_pyplot,
)


@pytest.mark.nlp
def test_pyplot_visualization(a_batch_text):
    # Just check that it doesn't fail with expected inputs.
    visualise_explanations_as_html(a_batch_text)


@pytest.mark.nlp
def test_pyplot_visualization(a_batch_text):
    # Just check that it doesn't fail with expected inputs.
    visualise_explanations_as_pyplot(a_batch_text)
