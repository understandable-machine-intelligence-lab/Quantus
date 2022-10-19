<!-- omit in toc -->
# Contribute to Quantus

Thank you for taking interest in contributions to Quantus!
We encourage you to contribute new features/metrics, optimisations, refactorings or report any bugs you may come across.
In this guide, you will get an overview of the workflow and best practices for contributing to Quantus.

**Questions.** If you have any developer-related questions, please [open an issue](https://github.com/understandable-machine-intelligence-lab/Quantus/issues/new/choose)
or write us at [anna.hedstroem@tu-berlin.de](mailto:anna.hedstroem@tu-berlin.de).

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [General Guide to Making Changes](#general-guide-to-making-changes)
  - [Development Installation](#development-installation)
  - [Branching](#branching)
  - [Code Style](#code-style)
  - [Unit Tests](#unit-tests)
  - [Before You Create a Pull Request](#before-you-create-a-pull-request)
  - [Pull Requests](#pull-requests)
- [Contributing a New Metric](#contributing-a-new-metric)
- [License](#license)

## Reporting Bugs

If you discover a bug, as a first step please check the existing [Issues](https://github.com/understandable-machine-intelligence-lab/Quantus//issues) to see if this bug has already been reported.
In case the bug has not been reported yet, please do the following:

- [Open an issue](https://github.com/understandable-machine-intelligence-lab/Quantus/issues/new/choose).
- Add a descriptive title to the issue and write a short summary of the problem.
- Adding more context, including reference to the problematic parts of the code, would be very helpful to us.

Once the bug is reported, our development team will try to address the issue as quickly as possible.

## General Guide to Making Changes

This is a general guide to contributing changes to Quantus. If you would like to add a new evaluation metric to Quantus, please refer to [Contributing a New Metric](#contributing-a-new-metric).
Before you start the development work, make sure to read our [documentation](https://quantus.readthedocs.io/) first.

### Development Installation

Make sure to install the latest version of Quantus from the main branch.
```bash
git clone https://github.com/understandable-machine-intelligence-lab/Quantus.git
cd quantus
pip install -r requirements_test.txt
pip install -e .
```

### Branching

Before you start making changes to the code, create a local branch from the latest version of `main`.

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

Make sure to add docstrings to every class, method and function that you add to the codebase. The docstring should include a description of all parameters and returns. Use the existing documentation as an example.
TODO: Automatic docstring generation.

### Before You Create a Pull Request

Before creating a PR, double-check that the following tasks are completed:

- Make sure that the latest version of the code from the `main` branch is merged into your working branch.
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

Once you are done with the changes:
- Create a [pull request](https://github.com/understandable-machine-intelligence-lab/Quantus/compare)
- Provide a summary of the changes you are introducing.
- In case you are resolving an issue, don't forget to link it.
- Add [annahedstroem](https://github.com/annahedstroem) as a reviewer.

## Contributing a New Metric

We always welcome extensions to our collection of evaluation metrics. This short description provides a guideline to introducing a new metric into Quantus. We strongly encourage you to take an example from already implemented metrics.

### Theoretical Foundations

Currently, we support six subgroups of evaluation metrics:
- Faithfulness
- Robustness
- Localisation
- Complexity
- Randomisation
- Axiomatic

See a more detailed description of those in [README](https://github.com/understandable-machine-intelligence-lab/Quantus#library-overview).
Identify which category your metric belongs to and create a Python file for your metric class in the respective folder in `quantus/metrics`.

### Metric Class
Every metric class inherits from the base `Metric` class: `quantus/metrics/base.py`. Importantly, Faithfulness and Robustness inherit not from the `Metric` class directly, but rather from its child `PerturbationMetric`.

A child metric can benefit from the following class methods:
- `__call__()`: Will call general_preprocess(), apply() on each instance and finally call custom_preprocess(). To use this method the child Metric needs to implement evaluate_instance().
- `general_preprocess()`: Prepares all necessary data structures for evaluation. Will call custom_preprocess() at the end.

The following methods are expected to be implemented in the metric class:
- `__init__()`: Initialize the metric.
- `__call__()`: Typically, calls `__call__()` in the base class.
- `evaluate_instance()`: Gets model and data for a single instance as input, returns evaluation result.

The following methods are optimal for implementation:
- `custom_preprocess()`: In case `general_preprocess()` from base class is not sufficient, additional preprocessing steps can be added here.
- `custom_postprocess()`: Additional postprocessing steps can be added here that is added on top of the resuling evaluation scores.

### Using Helpers

In the `quantus/helpers` folder, you might find functions relevant to your implementation. Use search function and go through the function docstrings to explore your options.

If you find yourself developing some functionality of more general scope, consider adding this code to a respective file, or creating a new module in `quantus/helpers`.

### Warnings
The `__init__()` method of a metric class typically call a warning that includes the following information:
- Metric name
- Sensitive parameters
- Proper citation of the source paper (!)

### Documenting a Metric
Declaration of a method class should be followed by:
- A detailed description of the metric
- References
- Assumptions

Otherwise, please remember to add a description for all parameters and returns of each new method/function, as well as a description of the purpose of the method/function itself.

## License
Please note that by contributing to the project you agree that it will be licensed under the [License](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/LICENSE).

## Questions

If you have any developer-related questions, please [open an issue](https://github.com/understandable-machine-intelligence-lab/Quantus/issues/new/choose)
or write us at [anna.hedstroem@tu-berlin.de](mailto:anna.hedstroem@tu-berlin.de).
