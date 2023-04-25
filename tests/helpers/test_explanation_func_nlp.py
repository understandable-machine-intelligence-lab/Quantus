from typing import List

import numpy as np
import pytest
from transformers_gradients.types import (
    NoiseGradConfig,
    NoiseGradPlusPlusConfig,
)

from quantus.functions.explanation_func import explain
from quantus.functions.nlp.explanation_func import ShapConfig
from quantus.functions.nlp.torch_explanation_func import LimeConfig


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
