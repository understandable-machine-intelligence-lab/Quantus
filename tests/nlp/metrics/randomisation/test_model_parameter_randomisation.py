import numpy as np
import pytest
from quantus.nlp import ModelParameterRandomisation


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert not (np.asarray(result) == np.NINF).all()


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_torch_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert not (np.asarray(result) == np.NINF).all()
