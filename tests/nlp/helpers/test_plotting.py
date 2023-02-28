import pytest
import numpy as np
from quantus.nlp.helpers.plotting import (
    visualise_explanations_as_html,
    visualise_explanations_as_pyplot,
    plot_token_flipping_experiment,
)


@pytest.mark.nlp
def test_pyplot_visualization(a_batch_text):
    # Just check that it doesn't fail with expected inputs.
    visualise_explanations_as_html(a_batch_text)


@pytest.mark.nlp
def test_pyplot_visualization(a_batch_text):
    # Just check that it doesn't fail with expected inputs.
    visualise_explanations_as_pyplot(a_batch_text)


@pytest.mark.nlp
def test_pyplot_token_prunning():
    scores = np.random.uniform(size=(8, 39))
    logits = np.random.uniform(size=(8,))
    # Just check that it doesn't fail with expected inputs.
    plot_token_flipping_experiment(scores, logits)
