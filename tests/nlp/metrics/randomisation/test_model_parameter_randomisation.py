import numpy as np
import pytest
from typing import Dict
from quantus.nlp import ModelParameterRandomisation
import os


@pytest.mark.nlp
@pytest.mark.skipif("SKIP_MPR" in os.environ)
@pytest.mark.randomisation
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
        (
            {"normalise": True},
            {"explain_func_kwargs": {"method": "GradNorm"}, "flatten_layers": True},
        ),
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = ModelParameterRandomisation(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, Dict)
    for i in result.values():
        # fmt: off
        assert not (i == np.NINF ).any()  # noqa
        assert not (i == np.PINF ).any()  # noqa
        assert not (i == np.NAN  ).any()  # noqa
        assert not (i == np.NZERO).any()
        assert not (i == np.PZERO).any()
        # fmt: on


@pytest.mark.nlp
@pytest.mark.skipif("SKIP_MPR" in os.environ)
@pytest.mark.randomisation
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
    assert isinstance(result, Dict)
    for i in result.values():
        # fmt: off
        assert not (i == np.NINF ).any()  # noqa
        assert not (i == np.PINF ).any()  # noqa
        assert not (i == np.NAN  ).any()  # noqa
        assert not (i == np.NZERO).any()
        assert not (i == np.PZERO).any()
        # fmt: on
