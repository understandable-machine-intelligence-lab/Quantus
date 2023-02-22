import pytest
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
        {"explain_func_kwargs": {"method": "GradXInput"}},
        {"explain_func_kwargs": {"method": "IntGrad"}},
    ],
}


@pytest.mark.nlp
@pytest.mark.evaluate_func
def test_tf_model(tf_sst2_model, sst2_dataset, capsys):
    # Just check that it doesn't fail with expected inputs.
    qn.evaluate(
        METRICS, model=tf_sst2_model, x_batch=sst2_dataset, call_kwargs=CALL_KWARGS
    )
