import numpy as np
import pytest

import quantus.nlp as qn


@pytest.mark.nlp
@pytest.mark.slow
def test_tf_model(tf_sst2_model, sst2_dataset):
    metrics = {
        "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5, disable_warnings=True),
        "Max-Sensitivity": qn.MaxSensitivity(nr_samples=5, disable_warnings=True),
        "RIS": qn.RelativeInputStability(nr_samples=5, disable_warnings=True),
        "RandomLogit": qn.RandomLogit(num_classes=2, disable_warnings=True),
        "TokenFlipping": qn.TokenFlipping(disable_warnings=True, abs=True),
        # "MPR": qn.ModelParameterRandomisation(disable_warnings=True),
    }

    call_kwargs = {
        "explain_func_kwargs": {"method": "GradXInput"},
        "batch_size": 6,
        "Max-Sensitivity": {
            "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
        },
        "RIS": [
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 5}},
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 7}},
        ],
    }
    # Just check that it doesn't fail with expected inputs.
    result = qn.evaluate(metrics, tf_sst2_model, sst2_dataset, call_kwargs=call_kwargs)
    result_ris = result["RIS"]
    # CHeck list of args returns list of scores
    assert isinstance(result_ris, list)
    assert len(result_ris) == 2
    assert isinstance(result_ris[0], np.ndarray)
    assert isinstance(result_ris[1], np.ndarray)


@pytest.mark.nlp
@pytest.mark.slow
def test_tf_model(torch_sst2_model, sst2_dataset):
    metrics = {
        "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5, disable_warnings=True),
        "Max-Sensitivity": qn.MaxSensitivity(nr_samples=5, disable_warnings=True),
        "RIS": qn.RelativeInputStability(nr_samples=5, disable_warnings=True),
        "RandomLogit": qn.RandomLogit(num_classes=2, disable_warnings=True),
        "TokenFlipping": qn.TokenFlipping(disable_warnings=True, abs=True),
        # "MPR": qn.ModelParameterRandomisation(disable_warnings=True),
    }

    call_kwargs = {
        "explain_func_kwargs": {"method": "GradXInput"},
        "batch_size": 6,
        "Max-Sensitivity": {
            "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
        },
        "RIS": [
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 5}},
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 7}},
        ],
    }
    # Just check that it doesn't fail with expected inputs.
    result = qn.evaluate(
        metrics, torch_sst2_model, sst2_dataset, call_kwargs=call_kwargs
    )
    result_ris = result["RIS"]
    # CHeck list of args returns list of scores
    assert isinstance(result_ris, list)
    assert len(result_ris) == 2
    assert isinstance(result_ris[0], np.ndarray)
    assert isinstance(result_ris[1], np.ndarray)
