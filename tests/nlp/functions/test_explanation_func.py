import numpy as np
import pytest
from typing import List
from quantus.nlp import explain, NoiseType


def unknown_token_baseline_function(_) -> np.ndarray:
    return np.load("tests/assets/nlp/unknown_token_embedding.npy")


@pytest.mark.nlp
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": False},
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
        {"method": "AttentionLast"},
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, kwargs):
    y_batch = tf_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(tf_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": False},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2, "m": 2},
        },
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2, "m": 2, "noise_type": "additive"},
        },
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
        {"method": "AttentionLast"},
    ],
)
def test_torch_model(emotion_model, emotion_dataset, kwargs):
    y_batch = emotion_model.predict(emotion_dataset).argmax(axis=-1)
    a_batch = explain(emotion_model, emotion_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
