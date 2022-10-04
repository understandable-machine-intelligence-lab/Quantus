<!-- omit in toc -->
# Contributing to Quantus

Thank you for taking interest in contributions to Quantus! We encourage you to contribute new features/metrics, optimize, refactor and report any bugs you may come across. In this guide you will get an overview of the workflow and best practices for contributing to Quantus.
<!-- omit in toc -->
## Table of Contents

- [Reporting Bugs](#get-started)
- [General Guide to Making Changes](#make-changes)
  - [Development Installation](#dev-installation)
  - [Code Style](#code-style)
  - [Unit Tests](#unit-tests)
  - [Checklist](#checklist)
  - [Before You Commit](#before-commit)
  - [Pull Requests](#pr)
- [Contributing a New Metric](#contributing-a-new-metric)
- [License](#license)




## Reporting Bugs

If you discover a bug, as a first step please check the existing [Issues](https://github.com/understandable-machine-intelligence-lab/Quantus//issues) to see if this bug has already been reported.
In case the bug has not been reported yet, please do the following:

- [Open an issue](https://github.com/understandable-machine-intelligence-lab/Quantus//issues/new).
- Add a descriptive title to the issue and write a short summary of the problem.
- Adding more context, including reference to the problematic parts of the code, would be very helpful to us.

Once the bug is reported, our development team will try to address the issue as quickly as possible.

## General Guide to Making Changes

This is a general guide to contributing changes to Quantus. If you would like to add a new evaluation metric to Quantus, please refer to [Contributing a New Metric](#contributing-a-new-metric).
Before you start the development work, make sure to read our [documentation](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/README.md) first.

### Development Installation
Make sure to install the latest version of Quantus from the main branch.
```bash
git clone https://github.com/understandable-machine-intelligence-lab/Quantus.git
cd quantus
pip install -r requirements_test.txt
pip install -e .
```

### Code Style
Code is written to follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) and for docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).
We use [flake8](https://pypi.org/project/flake8/) for quick style checks and [black](https://github.com/psf/black) for code formatting with a line-width of 88 characters per line.

### Unit Tests
Tests are written using [pytest](https://github.com/pytest-dev/pytest) and executed together with [codecov](https://github.com/codecov/codecov-action) for coverage reports.
To perform the tests, execute the following (make sure pytest is installed):
```bash
pytest
```
... alternatively, to get additionaly coverage details, run:
```bash
pytest --cov=. --cov-report term-missing
```

It is possible to limit the scope of testing to specific sections of the codebase, for example, only test the Faithfulness metrics:
```bash
pytest -m faithfulness -s
```
For a complete overview of the possible testing scopes, please refer to `pytest.ini`.
### Documentation
Make sure to add docstrings to every class, method and function that you add to the codebase.

### Checklist
Before creating a PR, double-check that the following tasks are completed:

- Run `black` to format source code:
```bash
black quantus/INSERT_YOUR_FILE_NAME.py
```
- Run `flake8` for quick style checks, e.g.: 
```bash
flake8 quantus/INSERT_YOUR_FILE_NAME.py
```
- Create `pytests` for new functionality (if needed) and add under `tests/` folder
- If the `pytests` include a new category of `@pytest.mark` then add that category with description to `pytest.ini`
- Make sure all unit tests are passed and the coverage level is maintained (we aim at ~100% code coverage for Quantus):
```bash
pytest tests -v --cov-report term --cov-report html:htmlcov --cov-report xml --cov=quantus
```

### Pull Requests
TODO

## Contributing a New Metric
TODO

## License
This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
