# Welcome to Quantus documentation!

**Quantus is an eXplainable AI toolkit for responsible evaluation of neural network explanations.**

[📑 Shortcut to paper!](https://arxiv.org/abs/2202.06861)

This documentation is complementary to Quantus repository's [README.md](https://github.com/understandable-machine-intelligence-lab/Quantus) and provides documentation
for how to install Quantus (**Installation**), how to contribute to the project (**Developer Documentation**) and on the interface (**API Documentation**).
For further guidance on how to best use the library, please read the user guidelines (**Guidelines**). More information about how to get started can be be found in the [README.md](https://github.com/understandable-machine-intelligence-lab/Quantus)
with various examples in [tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus/tree/main/tutorials) folder.

Quantus can be installed from [PyPI](https://pypi.org/project/quantus/)
(this way assumes that you have either `torch` or `tensorflow` already installed on your machine).

```setup
pip install quantus
```

For alternative ways to install Quantus package, read more under **Installation**.

```{toctree}
:caption: Installation
:maxdepth: 1

getting_started/installation
```

```{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/getting_started_example
```

```{toctree}
:caption: API Documentation
:maxdepth: 1

docs_api/modules
```

```{toctree}
:caption: Developer Documentation
:maxdepth: 1

docs_dev/contribution_guide
```

```{toctree}
:caption: User Guidelines
:maxdepth: 2

guidelines/guidelines_and_disclaimers
```


### Citation

If you find this toolkit or its companion paper
[**Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations**](https://arxiv.org/abs/2202.06861)
interesting or useful in your research, use following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2022quantus,
      title={Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations},
      author={Anna Hedström and
              Leander Weber and
              Dilyara Bareeva and
              Franz Motzkus and
              Wojciech Samek and
              Sebastian Lapuschkin and
              Marina M.-C. Höhne},
      year={2022},
      eprint={2202.06861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

When applying the individual metrics of Quantus, please make sure to also properly cite the work of the original authors.
