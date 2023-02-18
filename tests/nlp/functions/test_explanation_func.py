import numpy as np
import pytest
from typing import List
from quantus.nlp import explain, NoiseType
from tests.nlp.util import skip_on_apple_silicon


def unknown_token_baseline_function(_) -> np.ndarray:
    return np.load("tests/assets/nlp/unknown_token_embedding.npy")


@pytest.mark.nlp
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {"method": "NoiseGrad++", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "n": 2,
            "m": 2,
            "noise_type": NoiseType.additive,
        },
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
)
def test_tf_model(tf_distilbert_sst2_model, sst2_dataset, kwargs):
    y_batch = tf_distilbert_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(tf_distilbert_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {"method": "NoiseGrad++", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "n": 2,
            "m": 2,
            "noise_type": NoiseType.additive,
        },
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
)
def test_torch_model(torch_distilbert_sst2_model, sst2_dataset, kwargs):
    y_batch = torch_distilbert_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(torch_distilbert_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@skip_on_apple_silicon
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {"method": "NoiseGrad++", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "n": 2,
            "m": 2,
            "noise_type": NoiseType.additive,
        },
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
)
def test_keras_model(fnet_ag_news_model, sst2_dataset, kwargs):
    y_batch = fnet_ag_news_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(fnet_ag_news_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
