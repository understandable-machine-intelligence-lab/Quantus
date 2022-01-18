### How to run tests

Run all tests at once:

```pytest```

Run a subset of tests with e.g., localisation metrics (see available markers in the pytest.ini files):

```pytest -m localisation -s```

Run pytest with coverage:

```pytest tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov=quantus```