from typing import Union

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np

import quantus
from quantus.evaluation import evaluate, evaluate_nlp
from quantus.functions.explanation_func import explain
from quantus.functions.perturb_func import synonym_replacement, spelling_replacement

from quantus.metrics.complexity import Sparseness  # noqa
from quantus.metrics.robustness import MaxSensitivity  # noqa


@pytest.mark.evaluate_func
@pytest.mark.last
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


@pytest.mark.nlp
@pytest.mark.last
@pytest.mark.parametrize(
    "model, data",
    [
        (lazy_fixture("tf_sst2_model"), lazy_fixture("sst2_dataset")),
        (lazy_fixture("torch_sst2_model"), lazy_fixture("sst2_dataset")),
    ],
)
def test_evaluate_nlp(model, data):
    nlp_metrics = {
        # "MPR": quantus.ModelParameterRandomisation(disable_warnings=True),
        "RandomLogit": quantus.RandomLogit(disable_warnings=True),
        "TokenFlip": quantus.TokenFlipping(disable_warnings=True),
        "Avg-Sen": quantus.AvgSensitivity(disable_warnings=True, nr_samples=5, perturb_func=synonym_replacement),
        "Max-Sen": quantus.MaxSensitivity(disable_warnings=True, nr_samples=5, perturb_func=spelling_replacement),
        # "RIS": quantus.RelativeInputStability(disable_warnings=True, nr_samples=5),
        # "ROS": quantus.RelativeOutputStability(disable_warnings=True, nr_samples=5),
        # "RRS": quantus.RelativeRepresentationStability(disable_warnings=True, nr_samples=5)
    }

    call_kwargs = {
        "explain_func_kwargs": {"method": "GradXInput"},
        "batch_size": 8,
        "Max-Sen": {
            "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
        },
        "Avg-Sen": [
            {
                "explain_func_kwargs": {
                    "method": "IntGrad",
                }
            },
            {
                "explain_func_kwargs": {
                    "method": "GradXInput",
                }
            },
        ],
    }

    scores = evaluate_nlp(
        nlp_metrics,
        model,
        **data,
        call_kwargs={
            "explain_func": explain,
            **call_kwargs
        },
    )

    result_avg_sen = scores["Avg-Sen"]
    # Check list of args returns list of scores
    assert isinstance(result_avg_sen, list)
    assert len(result_avg_sen) == 2
    assert isinstance(result_avg_sen[0], np.ndarray)
    assert isinstance(result_avg_sen[1], np.ndarray)
