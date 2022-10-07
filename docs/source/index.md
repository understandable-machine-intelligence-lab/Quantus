# Welcome to Quantus documentation!

**Quantus is an eXplainable AI toolkit for responsible evaluation of neural network explanations.**

<p align="center">
  <img width="525" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/viz.png">
</p>
<p><small>
Figure: a) Simple qualitative comparison of XAI methods is often not sufficient to distinguish which
gradient-based method — Saliency, Integrated Gradients, GradientShap or FusionGrad
is preferred. With Quantus, we can obtain richer insights on how the methods compare b) by holistic
quantification on several evaluation criteria and c) by providing sensitivity analysis of how a single parameter
e.g. pixel replacement strategy of a faithfulness test influences the ranking of XAI methods.
</small></p>

[📑 Shortcut to paper!](https://arxiv.org/abs/2202.06861)


This documentation is complementary to Quantus repository's [README.md](https://github.com/understandable-machine-intelligence-lab/Quantus) and provides documentation
for how to install Quantus (**Installation**), how to contribute to the project (**Developer Documentation**) and on the interface (**API Documentation**).
For further guidance on what to think about when applying Quantus, please read the user guidelines (**Guidelines**).

Do you want to get started? Please have a look at our simple MNIST/torch/Saliency/IntGrad toy example (**Getting started**).
For more examples, check the [tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus/tree/main/tutorials) folder.

Quantus can be installed from [PyPI](https://pypi.org/project/quantus/)
(this way assumes that you have either `torch` or `tensorflow` already installed on your machine).

```setup
pip install quantus
```

For alternative ways to install Quantus, read more under **Installation**.

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

docs_dev/CONTRIBUTING.md
```

```{toctree}
:caption: Guidelines
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
