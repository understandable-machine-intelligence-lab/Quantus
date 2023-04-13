from typing import Union, Dict

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from pytest_mock import MockerFixture

import quantus
from quantus.evaluation import evaluate, evaluate_text_classification
from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import synonym_replacement, spelling_replacement
from quantus.metrics.complexity import Sparseness  # noqa
from quantus.metrics.robustness import MaxSensitivity  # noqa


@pytest.mark.order(-1)
@pytest.mark.evaluate_func
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
            lazy_fixture("load_1d_3ch_conv_model"),
            lazy_fixture("almost_uniform_1d_no_abatch"),
            {
                "explain_func_kwargs": {
                    "method": "Gradient",
                },
                "explain_func": explain,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**{'disable_warnings': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method']: a_batch}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "Saliency",
                },
                "explain_func": explain,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**{'disable_warnings': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method']: a_batch}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True,'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method']: a_batch}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "Gradient",
                },
                "explain_func": explain,
                "eval_metrics": "{'max-Sensitivity': MaxSensitivity(**{'disable_warnings': True,'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method']: a_batch}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method'] : params['explain_func']}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "None",
                "eval_xai_methods": "None",
                "call_kwargs": "{'0': {}}",
            },
            {"None": None},
        ),
        (
            lazy_fixture("load_mnist_model"),
            lazy_fixture("load_mnist_images"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method'] : params['explain_func_kwargs']}",
                "call_kwargs": "{'0': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_evaluate_func(
    model,
    data: np.ndarray,
    params: dict,
    expected: Union[float, dict, bool],
):
    x_batch, y_batch = data["x_batch"], data["y_batch"]
    explain = params["explain_func"]
    call_kwargs = params.get("call_kwargs", {})
    a_batch = explain(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **params["explain_func_kwargs"],
    )

    if "None" in expected:
        results = evaluate(
            metrics=eval(params["eval_metrics"]),
            xai_methods=eval(params["eval_xai_methods"]),
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            agg_func=np.mean,
            explain_func_kwargs=params["explain_func_kwargs"],
            call_kwargs=eval(params["call_kwargs"]),
        )
        assert results == None, "Test failed."

    results = evaluate(
        metrics=eval(params["eval_metrics"]),
        xai_methods=eval(params["eval_xai_methods"]),
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        agg_func=np.mean,
        explain_func_kwargs=params["explain_func_kwargs"],
        call_kwargs=eval(params["call_kwargs"]),
    )

    if "min" in expected and "max" in expected:
        assert (
            results[params["explain_func_kwargs"]["method"]][
                list(eval(params["eval_metrics"]).keys())[0]
            ][list(eval(params["call_kwargs"]).keys())[0]]
            >= expected["min"]
        ), "Test failed."
        assert (
            results[params["explain_func_kwargs"]["method"]][
                list(eval(params["eval_metrics"]).keys())[0]
            ][list(eval(params["call_kwargs"]).keys())[0]]
            <= expected["max"]
        ), "Test failed."


@pytest.fixture
def sst2_2_ragged_batches(sst2_dataset):
    # In real life we will have multiple batches, and it is very unlikely the dataset will be divisible
    # by batch size, so we simulate this case for this test.
    ds_1 = sst2_dataset.copy()
    ds_2 = sst2_dataset.copy()
    return {
        "x_batch": ds_1["x_batch"] + ds_2["x_batch"][:4],
        "y_batch": np.concatenate([ds_1["y_batch"], ds_2["y_batch"][:4]]),
    }


def assert_is_valid_score(scores):
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.floating
    assert scores.ndim == 1
    assert not np.isnan(scores).any()


@pytest.mark.order(-2)
@pytest.mark.nlp
@pytest.mark.parametrize(
    "model",
    [
        lazy_fixture("tf_sst2_model"),
        lazy_fixture("torch_sst2_model"),
    ],
    ids=["TF_model", "torch_model"],
)
def test_evaluate_nlp(
    model, sst2_2_ragged_batches, sst2_tokenizer, mocker: MockerFixture
):
    nlp_metrics = {
        "MPR": quantus.ModelParameterRandomisation(disable_warnings=True),
        "RandomLogit": quantus.RandomLogit(disable_warnings=True, num_classes=2),
        "TokenFlip": quantus.TokenFlipping(disable_warnings=True),
        "Avg-Sen": quantus.AvgSensitivity(
            disable_warnings=True, nr_samples=5, perturb_func=synonym_replacement
        ),
        "Max-Sen": quantus.MaxSensitivity(
            disable_warnings=True, nr_samples=5, perturb_func=spelling_replacement
        ),
        "RIS": quantus.RelativeInputStability(
            disable_warnings=True,
            nr_samples=5,
            return_nan_when_prediction_changes=False,
        ),
        "ROS": quantus.RelativeOutputStability(
            disable_warnings=True,
            nr_samples=5,
            return_nan_when_prediction_changes=False,
        ),
        "RRS": quantus.RelativeRepresentationStability(
            disable_warnings=True,
            nr_samples=5,
            return_nan_when_prediction_changes=False,
        ),
    }

    callback_stub = mocker.stub("callback_stub")
    scores = evaluate_text_classification(
        metrics=nlp_metrics,
        model=model,
        x_batch=sst2_2_ragged_batches["x_batch"],
        y_batch=sst2_2_ragged_batches["y_batch"],
        explain_func=quantus.explain,
        explain_func_kwargs=dict(method="GradXInput"),
        persist_callback=callback_stub,
        verbose=False,
        tokenizer=sst2_tokenizer,
    )

    assert callback_stub.call_count == 8
    for k in nlp_metrics:
        assert k in scores

    for k in nlp_metrics.keys():
        assert k in scores

    assert isinstance(scores["MPR"], Dict)
    for v in scores["MPR"].values():
        assert_is_valid_score(v)

    scores.pop("MPR")

    token_flip_scores = scores.pop("TokenFlip")

    assert_is_valid_score(token_flip_scores)

    for v in scores.values():
        assert_is_valid_score(v)
