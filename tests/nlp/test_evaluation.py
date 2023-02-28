import numpy as np
import pytest
import quantus.nlp as qn


@pytest.mark.order("last")
@pytest.mark.nlp
@pytest.mark.evaluate_func
def test_tf_model(tf_sst2_model, sst2_dataset):
    metrics = {
        "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5),
        "Max-Sensitivity": qn.MaxSensitivity(nr_samples=5),
        "Local Lipschitz Estimate": qn.LocalLipschitzEstimate(nr_samples=5),
        "RIS": qn.RelativeInputStability(nr_samples=5),
        "ROS": qn.RelativeOutputStability(nr_samples=5),
        "RRS": qn.RelativeRepresentationStability(nr_samples=5),
        # "ModelParameterRandomisation": qn.ModelParameterRandomisation(),
        "RandomLogit": qn.RandomLogit(num_classes=2),
        "TokenFlipping": qn.TokenFlipping(),
    }

    call_kwargs = {
        "Max-Sensitivity": {
            "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
        },
        "Local Lipschitz Estimate": [
            {"explain_func_kwargs": {"method": "GradXInput"}},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ],
    }
    # Just check that it doesn't fail with expected inputs.
    qn.evaluate(metrics, tf_sst2_model, sst2_dataset, call_kwargs=call_kwargs)


@pytest.mark.nlp
@pytest.mark.evaluate_func
def test_list_of_args_return_list_of_scores(tf_sst2_model, sst2_dataset):
    metrics = {
        "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5),
    }
    call_kwargs = {
        "Avg-Sesnitivity": [
            {"explain_func_kwargs": {"method": "GradXInput"}},
            {"explain_func_kwargs": {"method": "IntGrad"}},
        ],
    }
    result = qn.evaluate(metrics, tf_sst2_model, sst2_dataset, call_kwargs=call_kwargs)

    result = result["Avg-Sesnitivity"]
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
