import numpy as np
import pytest
from quantus.nlp import RandomLogit


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "num_classes": 2},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs):
    metric = RandomLogit(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    # fmt: off
    assert not (result == np.NINF ).any() # noqa
    assert not (result == np.PINF ).any() # noqa
    assert not (result == np.NAN  ).any() # noqa
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    # fmt: on


@pytest.mark.nlp
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs",
    [
        (
            {"normalise": True, "num_classes": 6},
            {"explain_func_kwargs": {"method": "GradNorm"}},
        ),
    ],
)
def test_torch_model(emotion_model, emotion_dataset, init_kwargs, call_kwargs):
    metric = RandomLogit(**init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    # fmt: off
    assert not (result == np.NINF ).any()  # noqa
    assert not (result == np.PINF ).any()  # noqa
    assert not (result == np.NAN  ).any()  # noqa
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    # fmt: on
