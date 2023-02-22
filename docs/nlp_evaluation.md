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
- return sample correlation for MPR.
- Layer selection for RRS.
- LRP-based explanation methods.
- `softmax` check's
- docstring for metrics
- tests for invalid inputs/arguments.
- tests for `return_nan_when_prediction_changes`