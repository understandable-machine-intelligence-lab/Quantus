from typing import List, Callable

import numpy as np
import pytest
import tensorflow as tf

from transformers_gradients.config import (
    IntGradConfig,
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
)
from quantus.functions.explanation_func import explain
from quantus.functions.nlp.explanation_func import ShapConfig
from quantus.functions.nlp.lime import LimeConfig
from quantus.helpers.tf_utils import is_xla_compatible_platform


def unk_token_baseline_func() -> Callable:
    unknown_token = tf.constant(np.load("tests/assets/unknown_token_embedding.npy"))

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def unk_token_baseline(x):
        return unknown_token

    return unk_token_baseline


@pytest.mark.nlp
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad", "config": IntGradConfig(batch_interpolated_inputs=False)},
        # It is not really slow, but rather running it with xdist can crash runner with OOM.
        pytest.param({"method": "IntGrad"}, marks=pytest.mark.slow),
        {
            "method": "IntGrad",
            "config": IntGradConfig(
                baseline_fn=unk_token_baseline_func(), batch_interpolated_inputs=False
            ),
        },
        {
            "method": "NoiseGrad",
            "config": NoiseGradConfig(n=2, explain_fn="GradXInput"),
        },
        {
            "method": "NoiseGrad++",
            "config": NoiseGradPlusPlusConfig(
                n=2, m=2, explain_fn="GradNorm", noise_fn=lambda a, b: a + b
            ),
        },
        {"method": "LIME", "config": LimeConfig(num_samples=5)},
        {"method": "SHAP", "config": ShapConfig(max_evals=5)},
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
def test_tf_model(tf_sst2_model_wrapper, sst2_dataset, kwargs):
    a_batch = explain(
        tf_sst2_model_wrapper,
        inputs=sst2_dataset["x_batch"],
        targets=sst2_dataset["y_batch"],
        **kwargs,
    )
    assert len(a_batch) == 8
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        # assert isinstance(scores, (np.ndarray, tf.Tensor))
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
            "config": NoiseGradConfig(n=2),
        },
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "config": NoiseGradPlusPlusConfig(n=2, m=2),
        },
        {"method": "LIME", "config": LimeConfig(num_samples=5)},
        {"method": "SHAP", "config": ShapConfig(max_evals=5)},
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
def test_torch_model(torch_sst2_model_wrapper, sst2_dataset, kwargs):
    a_batch = explain(
        torch_sst2_model_wrapper,
        inputs=sst2_dataset["x_batch"],
        targets=sst2_dataset["y_batch"],
        **kwargs,
    )
    assert len(a_batch) == 8
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
