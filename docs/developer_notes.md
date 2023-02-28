#### Step 1. Create API docs!
Read more here: https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html and here: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
```bash
$ cd docs
$ make clean
$ make rst
$ make html
```

#### Step 2. View edits and make changes accordingly.
http://localhost:63342/Projects/quantus/docs/build/html/index.html

A copy is made of CONTRIBUTING.md to docs_dev/CONTRIBUTING.md. To avoid any inconsistencies, edit in CONTRIBUTING.md and overwrite in docs_dev/CONTRIBUTING.md.


# NLP evaluation highlights

#### Implemented
- Sensitivity metrics.
- Relative stability metrics.
- Randomisation metrics.
- XAI methods.

#### Major API differences
- `x_batch` is a `List[str]`
- `y_batch` is optional.
- `__init__` accepts keyword-only arguments.
- no `s_batch`.


#### Not yet implemented
- `softmax` check's
- `return_aggregate` behaviour
- Proper handling for invalid inputs/arguments.
- `return_nan_when_prediction_changes` behaviour.
- NLP tasks beside sentiment analysis.
- LRP-based XAI method's for TF.
