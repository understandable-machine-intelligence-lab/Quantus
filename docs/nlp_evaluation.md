### NLP evaluation highlights

# Implemented
- Sensitivity metrics.
- Relative stability metrics.
- Randomisation metrics.

# Major API differences
- `x_batch` is a `List[str]`
- `y_batch` is optional.
- `__init__` accepts keyword-only arguments.
- no `s_batch`.


# Not yet implemented
- `return_aggregate` for robustness metrics.
- `softmax` check's
- tests for invalid inputs/arguments.
- tests for `return_nan_when_prediction_changes`
- NLP tasks beside sentiment analysis

# TODO
- Keras model
- `return_aggregate`
- LRP-based explanation methods.
- return sample correlation for MPR.
- docstring for metrics
- multi-dispatch instead of method with different names.