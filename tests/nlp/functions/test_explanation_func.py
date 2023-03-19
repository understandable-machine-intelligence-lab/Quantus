from typing import List

import numpy as np
import pytest
import tensorflow as tf
from quantus.nlp import (
    explain,
    IntGradConfig,
    tf_function,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
)


@tf_function
def unk_token_baseline(x):
    unknown_token = np.load("tests/assets/nlp/unknown_token_embedding.npy")
    return unknown_token


@pytest.mark.nlp
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad", "config": IntGradConfig(batch_interpolated_inputs=False)},
        # It is not really slow, but rather running it with xdist can crash runner with OOM.
        pytest.param({"method": "IntGrad"}, marks=pytest.mark.slow),
        {"method": "IntGrad", "config": IntGradConfig(baseline_fn=unk_token_baseline)},
        {
            "method": "NoiseGrad",
            "config": NoiseGradConfig(n=2, explain_fn="GradXInput"),
        },
        {
            "method": "NoiseGrad++",
            "config": NoiseGradPlusPlusConfig(
                n=2, m=2, explain_fn="GradNorm", noise_fn="additive"
            ),
        },
        {"method": "LIME", "num_samples": 5},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad",
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
        assert isinstance(scores, (np.ndarray, tf.Tensor))
        assert len(scores.shape) == 1


@pytest.mark.nlp
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {
            "method": "NoiseGrad",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2},
        },
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2, "m": 2},
        },
        {"method": "LIME", "num_samples": 5},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad",
        "NoiseGrad",
        "NoiseGrad++",
        "LIME",
        "SHAP",
    ],
)
def test_torch_model(torch_sst2_model, sst2_dataset, kwargs):
    y_batch = torch_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(torch_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
