import numpy as np
import pytest
from typing import List
from pytest_lazyfixture import lazy_fixture
from tests.nlp.utils import skip_on_apple_silicon
from quantus.nlp import explain, NoiseType


def unknown_token_baseline_function(_) -> np.ndarray:
    return np.load("tests/assets/nlp/unknown_token_embedding.npy")


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
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
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad++",
        "NoiseGrad++ additive noise",
        "LIME",
        "SHAP",
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
@pytest.mark.keras_nlp
@skip_on_apple_silicon
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
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
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad++",
        "NoiseGrad++ additive noise",
        "LIME",
        "SHAP",
    ],
)
def test_keras_model(fnet_keras, ag_news_dataset, kwargs):
    y_batch = fnet_keras.predict(ag_news_dataset).argmax(axis=-1)
    a_batch = explain(fnet_keras, ag_news_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "model, dataset, kwargs",
    [
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "GradNorm"},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "GradXInput"},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "IntGrad"},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "IntGrad", "batch_interpolated_inputs": False},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {
                "method": "NoiseGrad++",
                "explain_fn": "GradXInput",
                "init_kwargs": {"n": 2, "m": 2},
            },
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {
                "method": "NoiseGrad++",
                "explain_fn": "GradXInput",
                "init_kwargs": {"n": 2, "m": 2, "noise_type": "additive"},
            },
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        ),
        (
            lazy_fixture("emotion_model"),
            lazy_fixture("emotion_dataset"),
            {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "GradNorm"},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "GradXInput"},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "IntGrad"},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "IntGrad", "batch_interpolated_inputs": False},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {
                "method": "NoiseGrad++",
                "explain_fn": "GradXInput",
                "init_kwargs": {"n": 2, "m": 2},
            },
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {
                "method": "NoiseGrad++",
                "explain_fn": "GradXInput",
                "init_kwargs": {"n": 2, "m": 2, "noise_type": "additive"},
            },
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        ),
        (
            lazy_fixture("torch_fnet"),
            lazy_fixture("sst2_dataset"),
            {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
        ),
    ],
    ids=[
        "emotion_model -> GradNorm",
        "emotion_model -> GradXInput",
        "emotion_model -> IntGrad iterative",
        "emotion_model -> IntGrad batched",
        "emotion_model -> IntGrad [UNK] baseline",
        "emotion_model -> NoiseGrad++",
        "emotion_model -> NoiseGrad++ additive noise",
        "emotion_model -> LIME",
        "emotion_model -> SHAP",
        "fnet -> GradNorm",
        "fnet -> GradXInput",
        "fnet -> IntGrad iterative",
        "fnet -> IntGrad batched",
        "fnet -> IntGrad [UNK] baseline",
        "fnet -> NoiseGrad++",
        "fnet -> NoiseGrad++ additive noise",
        "fnet -> LIME",
        "fnet -> SHAP",
    ],
)
def test_torch_model(model, dataset, kwargs):
    y_batch = model.predict(dataset).argmax(axis=-1)
    a_batch = explain(model, dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
