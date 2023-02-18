from typing import Dict

import numpy as np

import quantus.nlp as qn

METRICS = {
    "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5),
    "Max-Sensitivity": qn.MaxSensitivity(nr_samples=5),
    "Local Lipschitz Estimate": qn.LocalLipschitzEstimate(nr_samples=5),
    "RIS": qn.RelativeInputStability(nr_samples=5),
    "ROS": qn.RelativeOutputStability(nr_samples=5),
    "RRS": qn.RelativeRepresentationStability(nr_samples=5),
    "ModelParameterRandomisation": qn.ModelParameterRandomisation(),
    "RandomLogit": qn.RandomLogit(num_classes=2),
    "TokenFlipping": qn.TokenFlipping(),
}

CALL_KWARGS = {
    "Max-Sensitivity": {
        "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
    },
    "Local Lipschitz Estimate": [
        qn.MetricCallKwargs("var1", {"explain_func_kwargs": {"method": "GradXInput"}}),
        qn.MetricCallKwargs("var2", {"explain_func_kwargs": {"method": "IntGrad"}}),
    ],
}


def test_tf_model(tf_sst2_model, sst2_dataset, capsys):
    result = qn.evaluate(
        METRICS, model=tf_sst2_model, x_batch=sst2_dataset, call_kwargs=CALL_KWARGS
    )
    for value in result.values():
        if isinstance(value, Dict):
            for i in value.values():
                assert isinstance(i, np.ndarray)
        else:
            assert isinstance(value, np.ndarray)
