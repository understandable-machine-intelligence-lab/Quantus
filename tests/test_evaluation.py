from typing import Union

import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
from quantus.evaluation import evaluate
from quantus.functions.explanation_func import explain

from quantus.metrics.complexity import Sparseness, Complexity
from quantus.metrics.robustness import MaxSensitivity


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
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method'] : params['explain_func_kwargs']}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "explain_func_kwargs": {
                    "method": "IntegratedGradients",
                },
                "explain_func": explain,
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,}),"
                "'Complexity': Complexity(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{params['explain_func_kwargs']['method'] : params['explain_func_kwargs']}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{'Saliency' : np.random.uniform(0, 0.01, size=(8, 28, 28, 1)),"
                "'IntegratedGradients' : np.random.uniform(0.1, 1.0, size=(8, 28, 28, 1))}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{'SaliencyUni' : np.random.uniform(0, 0.01, size=(8, 28, 28, 1)),"
                "'Saliency' : {'xai_lib': 'captum'}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{'SaliencyUni' : np.random.uniform(0, 0.01, size=(8, 28, 28, 1)),"
                "'Saliency' : {'xai_lib': 'captum'}}",
                "call_kwargs": "{'run_1': {}}",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{'SaliencyUni' : np.random.uniform(0, 0.01, size=(8, 28, 28, 1)),"
                "'Saliency' : {'xai_lib': 'captum'}}",
                "call_kwargs": "{'run_1': {}}",
                "return_as_df": "True",
            },
            {"min": 0.0, "max": 1.0},
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {
                "eval_metrics": "{'Sparseness': Sparseness(**{'disable_warnings': True, 'normalise': True,})}",
                "eval_xai_methods": "{'SaliencyUni' : np.random.uniform(0, 0.01, size=(8, 28, 28, 1)),"
                "'Saliency' : {'xai_lib': 'captum'}}",
                "call_kwargs": "{'run_1': {'batch_size': 4}, 'run_2': {'batch_size': 8}}",
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
    explain = params.get("explain_func", None)
    explain_func_kwargs = params.get("explain_func_kwargs", None)
    return_as_df = params.get("return_as_df", None)

    call_kwargs = params.get("call_kwargs", None)
    try:
        call_kwargs = eval(params["call_kwargs"])
    except Exception as e:
        print(e)

    if explain_func_kwargs is not None:
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )

    if "None" in expected:
        results = evaluate(
            metrics=eval(params["eval_metrics"]),
            xai_methods=eval(params["eval_xai_methods"]),
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            agg_func=np.mean,
            explain_func_kwargs=explain_func_kwargs,
            call_kwargs=call_kwargs,
            return_as_df=return_as_df,
        )
        assert results == None, "Test failed."

    else:
        results = evaluate(
            metrics=eval(params["eval_metrics"]),
            xai_methods=eval(params["eval_xai_methods"]),
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            agg_func=np.mean,
            explain_func_kwargs=explain_func_kwargs,
            call_kwargs=call_kwargs,
            return_as_df=return_as_df,
        )

        if return_as_df:
            assert isinstance(results, pd.DataFrame), "Test failed."
        else:
            assert isinstance(results, dict), "Test failed."

        # Slice from result dictionary.
        index_metric, index_method = 0, 0
        xai_method = list(eval(params["eval_xai_methods"]).keys())[index_method]
        metric = list(eval(params["eval_metrics"]).keys())[index_metric]

        if call_kwargs is not None:
            if len(call_kwargs) > 1:
                index_call = 0
                call_run = list(call_kwargs.keys())[index_call]
                results[xai_method][metric][call_run] >= expected["min"], "Test failed."
                results[xai_method][metric][call_run] <= expected["max"], "Test failed."
            return None
        assert results[xai_method][metric] >= expected["min"], "Test failed."
        assert results[xai_method][metric] <= expected["max"], "Test failed."
