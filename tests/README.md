### How to run tests

Run all unit tests at once on all available CPU cores:

```shell
pytest -n auto
```

Run a subset of tests with e.g., localisation metrics (see available markers in the pytest.ini files):

```shell
pytest -m localisation -s
```

Run pytest with coverage:

```shell
tox run -r coverage
```

### Integration tests
These tests verify, that quantus is usable without all optional dependencies installed.

Run integration tests in isolated environment:
- base TensorFlow installation
```shell
tox -e run tf_only
```
- base Torch installation
```shell
tox -e run torch_only
```

- TensorFlow + NLP installation
```shell
tox -e run tf_nlp
```

- Torch + NLP installation
```shell
tox -e run torch_nlp
```
