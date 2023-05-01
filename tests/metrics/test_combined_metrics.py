import pytest
import tensorflow as tf
import transformers_gradients
from transformers_gradients import update_config
from transformers_gradients.assertions import assert_numerics

import quantus


def tf_fro_norm(arr):
    if tf.rank(arr) == 3:
        axis = (1, 2)
    else:
        axis = None
    return tf.linalg.norm(arr, ord=1, axis=axis)


@pytest.fixture()
def return_scores_only():
    update_config(return_raw_scores=True)
    yield
    update_config(return_raw_scores=False)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "perturb_func, perturb_func_kwargs",
    [(quantus.spelling_replacement, dict(k=3)), (quantus.gaussian_noise, {})],
    ids=["spelling_replacement", "gaussian_noise"],
)
def test_avg_and_max_sensitivity(
    tf_sst2_model,
    sst2_tokenizer,
    sst2_dataset,
    return_scores_only,
    perturb_func,
    perturb_func_kwargs,
):
    metric = quantus.AvgAndMaxSensitivity(
        perturb_func=perturb_func,
        perturb_func_kwargs=perturb_func_kwargs,
        normalise=True,
        normalise_func=transformers_gradients.normalize_sum_to_1,
        similarity_func=tf.math.subtract,
        norm_numerator=tf_fro_norm,
        norm_denominator=tf_fro_norm,
        nr_samples=5,
    )
    scores = metric(
        tf_sst2_model,
        sst2_dataset["x_batch"],
        sst2_dataset["y_batch"],
        explain_func=quantus.explain,
        explain_func_kwargs=dict(method="NoiseGrad", config=dict(n=2)),
        tokenizer=sst2_tokenizer,
    )

    assert isinstance(scores, dict)
    max_val = scores["max"]
    avg_val = scores["avg"]

    tf.debugging.assert_rank(max_val, 1)
    tf.debugging.assert_rank(avg_val, 1)

    assert len(max_val) == len(sst2_dataset["x_batch"])
    assert len(avg_val) == len(sst2_dataset["y_batch"])

    assert_numerics(max_val)
    assert_numerics(avg_val)
    tf.debugging.assert_greater_equal(max_val, avg_val)

    assert tf.reduce_all(max_val != 0.0)
    assert tf.reduce_all(avg_val != 0.0)


@pytest.mark.robustness
@pytest.mark.parametrize(
    "perturb_func, perturb_func_kwargs",
    [(quantus.spelling_replacement, dict(k=3)), (quantus.gaussian_noise, {})],
    ids=["spelling_replacement", "gaussian_noise"],
)
def test_relative_stability(
    tf_sst2_model,
    sst2_tokenizer,
    sst2_dataset,
    return_scores_only,
    perturb_func,
    perturb_func_kwargs,
):
    metric = quantus.CombinedRelativeStability(
        perturb_func=perturb_func,
        perturb_func_kwargs=perturb_func_kwargs,
        normalise=True,
        normalise_func=transformers_gradients.normalize_sum_to_1,
        nr_samples=5,
    )
    scores = metric(
        tf_sst2_model,
        sst2_dataset["x_batch"],
        sst2_dataset["y_batch"],
        explain_func=quantus.explain,
        explain_func_kwargs=dict(method="NoiseGrad", config=dict(n=2)),
        tokenizer=sst2_tokenizer,
    )

    assert isinstance(scores, dict)
    ris = scores["ris"]
    ros = scores["ros"]
    rrs = scores["rrs"]

    tf.debugging.assert_rank(ris, 1)
    tf.debugging.assert_rank(ros, 1)
    tf.debugging.assert_rank(rrs, 1)

    assert len(ris) == len(sst2_dataset["x_batch"])
    assert len(ros) == len(sst2_dataset["y_batch"])
    assert len(rrs) == len(sst2_dataset["y_batch"])

    assert_numerics(ris)
    assert_numerics(ros)
    assert_numerics(rrs)
