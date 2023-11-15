### How to run tests

Run all tests for all supported python versions, execute:

```shell
python3 -m tox run
```

... or to run all testing environments in parallel, execute:

```shell
python3 -m tox run-parallel
```

To list all configured test environments, run

```shell
python3 -m tox list
```

To run, e.g., only test for python3.8, run:

```shell
python3 -m tox run -e py38
```

If you need to provide additional CLI argument, they must follow after `--`, e.g., in this case,
we will split test execution between cpu cores using [pytest-xdist](https://github.com/pytest-dev/pytest-xdist):

```shell
python3 -m tox run -e py310 -- -n auto
```

Run a subset of tests with e.g., localisation metrics (see available markers in the pytest.ini files):

```shell
python3 -m tox run -- -m localisation -s
```

Run pytest with coverage:

```shell
python3 -m tox run -e coverage
```

Run type checking using [mypy](https://github.com/python/mypy)

```shell
python3 -m tox run -e type
```

You can run all testing environments in parallel using multiprocessing by running:
```shell
python3 -m tox run-parallel
```