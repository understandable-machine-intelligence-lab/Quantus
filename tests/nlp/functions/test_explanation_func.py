from typing import List
import numpy as np
import pytest
from tests.nlp.utils import skip_on_apple_silicon
from quantus.nlp import explain, TorchHuggingFaceTextClassifier


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


@skip_on_apple_silicon
@pytest.mark.nlp
@pytest.mark.keras_nlp_model
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
        {"method": "NoiseGrad++", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "NoiseGrad++",
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
@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2, "m": 2},
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
        "LIME",
        "SHAP",
    ],
)
def test_torch_emotion_model(emotion_model, emotion_dataset, kwargs):
    y_batch = emotion_model.predict(emotion_dataset).argmax(axis=-1)
    a_batch = explain(emotion_model, emotion_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.explain_func
@pytest.mark.pytorch_model
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
        # {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad++",
        # "LIME",
        "SHAP",
    ],
)
def test_torch_fnet_model(torch_fnet, sst2_dataset, kwargs):
    y_batch = torch_fnet.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(torch_fnet, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize(
    "kwargs",
    [
        # {"method": "LRP-Ali"},
        # {"method": "LRP-Ali", "detach_layernorm": False},
        # {"method": "LRP-Ali", "detach_kq": False},
        # {"method": "LRP-Ali", "detach_mean": False},
        {"method": "LRP-Chefer"},
    ],
    # ids=[
    #    "Ali",
    #    "Ali: detach_layernorm=False",
    #    "Ali: detach_kq=False",
    #    "Ali: detach_mean",
    # ],
)
def test_bert_lrp_torch(sst2_dataset, kwargs):
    model = TorchHuggingFaceTextClassifier.from_pretrained(
        "gchhablani/bert-base-cased-finetuned-sst2"
    )
    y_batch = model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
