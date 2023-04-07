from __future__ import annotations

import functools

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import typo_replacement, spelling_replacement, synonym_replacement
from quantus.metrics.robustness import (
    RelativeInputStability,
    RelativeOutputStability,
    RelativeRepresentationStability,
)

# fmt: off
RIS_CONSTRUCTOR = functools.partial(RelativeInputStability, nr_samples=5, disable_warnings=True)
ROS_CONSTRUCTOR = functools.partial(RelativeOutputStability, nr_samples=5, disable_warnings=True)
RRS_CONSTRUCTOR = functools.partial(RelativeRepresentationStability, nr_samples=5, disable_warnings=True)
# fmt: on


relative_stability_tests = pytest.mark.parametrize(
    "model,data,init_kwargs,call_kwargs",
    [
        pytest.param(
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {},
            {"explain_func_kwargs": {"method": "GradientShap"}},
            id="mnist",
        ),
        pytest.param(
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "abs": True,
                "normalise": True,
                "return_aggregate": True,
            },
            {},
            id="mnist_aggregate",
        ),
        pytest.param(
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {},
            {"explain_func_kwargs": {"method": "IntegratedGradients"}},
            id="tf_mnist",
        ),
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d"),
            {},
            {},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("almost_uniform_2d_no_abatch"),
            {},
            {},
        ),
        # ------------ NLP ------------
        pytest.param(
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {"perturb_func": typo_replacement},
            {},
            marks=[pytest.mark.slow, pytest.mark.nlp],
            id="tf_nlp_plain_text",
        ),
        pytest.param(
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {
                "perturb_func": typo_replacement,
            },
            {},
            marks=[pytest.mark.slow, pytest.mark.nlp],
            id="torch_nlp_plain_text",
        ),
        pytest.param(
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {},
            {},
            marks=[pytest.mark.slow, pytest.mark.nlp],
            id="tf_nlp_latent",
        ),
        pytest.param(
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_dataset"),
            {},
            {},
            marks=[pytest.mark.slow, pytest.mark.nlp],
            id="torch_nlp_latent",
        ),
    ],
)


@pytest.mark.robustness
@relative_stability_tests
def test_relative_input_stability(model, data, init_kwargs, call_kwargs):
    x_batch = data["x_batch"]
    ris = RIS_CONSTRUCTOR(return_nan_when_prediction_changes=False, **init_kwargs)
    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=data["y_batch"],
        explain_func=explain,
        
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == ()
    else:
        assert result.shape[0] == len(x_batch)


@pytest.mark.robustness
@relative_stability_tests
def test_relative_output_stability(model, data, init_kwargs, call_kwargs):
    x_batch = data["x_batch"]
    ris = ROS_CONSTRUCTOR(return_nan_when_prediction_changes=False, **init_kwargs)

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=data["y_batch"],
        explain_func=explain,
        
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == ()
    else:
        assert result.shape[0] == len(x_batch)


@pytest.mark.robustness
@relative_stability_tests
def test_relative_representation_stability(
    model, data, init_kwargs, call_kwargs, torch_device
):
    x_batch = data["x_batch"]
    ris = RRS_CONSTRUCTOR(return_nan_when_prediction_changes=False, **init_kwargs)
    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=data["y_batch"],
        explain_func=explain,
        
        **call_kwargs,
    )
    result = np.asarray(result)
    assert (result != np.nan).all()
    if init_kwargs.get("return_aggregate", False):
        assert result.shape == ()
    else:
        assert result.shape[0] == len(x_batch)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "metric, model, data",
    [
        (
            RIS_CONSTRUCTOR,
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
        ),
        (
            ROS_CONSTRUCTOR,
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
        ),
        (
            RRS_CONSTRUCTOR,
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
        ),
        # ------- NLP --------
        (
            functools.partial(RIS_CONSTRUCTOR, perturb_func=spelling_replacement),
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
        ),
        (
            functools.partial(ROS_CONSTRUCTOR, perturb_func=spelling_replacement),
            lazy_fixture("tf_sst2_model"),
            lazy_fixture("sst2_dataset"),
        ),
        (
            functools.partial(RRS_CONSTRUCTOR, perturb_func=synonym_replacement),
            lazy_fixture("torch_sst2_model"),
            lazy_fixture("sst2_dataset"),
        ),
    ],
    ids=["RIS", "ROS", "RRS", "RIS_nlp", "ROS_nlp", "RRS_nlp"],
)
def test_return_nan(
    metric, model, data, mock_prediction_changed, torch_device
):
    rs = metric(return_nan_when_prediction_changes=True)
    result = rs(
        model=model,
        x_batch=data["x_batch"],
        y_batch=data["y_batch"],
        explain_func=explain,
        device=torch_device
    )
    assert np.isnan(result).all()
