import numpy as np
import pytest
from tests.nlp.markers import skip_on_apple_silicon
from quantus.nlp.metrics.faithfullness.token_flipping import TokenFlipping


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 39)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 39),
        ),
        (
            {"abs": True, "return_auc_per_sample": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8,),
        ),
    ],
    ids=["pruning", "activation", "AUC"],
)
def test_tf_model(
    tf_sst2_model, sst2_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(tf_sst2_model, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    assert result.shape == expected_shape


@pytest.mark.nlp
@skip_on_apple_silicon
@pytest.mark.keras_nlp_model
@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 40)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 40),
        ),
        (
            {"abs": True, "return_auc_per_sample": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8,),
        ),
    ],
    ids=["pruning", "activation", "AUC"],
)
def test_keras_model(
    fnet_keras, ag_news_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(fnet_keras, ag_news_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    assert result.shape == expected_shape


@pytest.mark.nlp
@pytest.mark.pytorch_model
@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 44)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 44),
        ),
        (
            {"abs": True, "return_auc_per_sample": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8,),
        ),
    ],
    ids=["pruning", "activation", "AUC"],
)
def test_emotion_torch_model(
    emotion_model, emotion_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(emotion_model, emotion_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)

    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()

    assert result.shape == expected_shape


@pytest.mark.nlp
@pytest.mark.pytorch_model
@pytest.mark.faithfulness
@pytest.mark.parametrize(
    "init_kwargs, call_kwargs, expected_shape",
    [
        ({"normalise": True}, {"explain_func_kwargs": {"method": "GradNorm"}}, (8, 42)),
        (
            {"normalise": True, "task": "activation"},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8, 42),
        ),
        (
            {"abs": True, "return_auc_per_sample": True},
            {"explain_func_kwargs": {"method": "GradNorm"}},
            (8,),
        ),
    ],
    ids=["pruning", "activation", "AUC"],
)
def test_fnet_torch_model(
    torch_fnet, sst2_dataset, init_kwargs, call_kwargs, expected_shape
):
    metric = TokenFlipping(**init_kwargs)
    result = metric(torch_fnet, sst2_dataset, **call_kwargs)
    assert isinstance(result, np.ndarray)
    assert not (result == np.NINF).any()
    assert not (result == np.PINF).any()
    assert not (result == np.NAN).any()
    assert not (result == np.NZERO).any()
    assert not (result == np.PZERO).any()
    assert result.shape == expected_shape
