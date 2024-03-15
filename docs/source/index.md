# Welcome to Quantus documentation!

**Quantus is an eXplainable AI toolkit for responsible evaluation of neural network explanations.**

<p align="center">
  <img width=“400” src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/viz.png">
</p>
<p><small>
Figure: a) Simple qualitative comparison of XAI methods is often not sufficient to distinguish which
gradient-based method — Saliency, Integrated Gradients, GradientShap or FusionGrad
is preferred. With Quantus, we can obtain richer insights on how the methods compare b) by holistic
quantification on several evaluation criteria and c) by providing sensitivity analysis of how a single parameter,
e.g., pixel replacement strategy of a faithfulness test influences the ranking of XAI methods. <a href="https://arxiv.org/abs/2202.06861">📑 Shortcut to paper!</a>
</small></p>


This documentation is complementary to the [README.md](https://github.com/understandable-machine-intelligence-lab/Quantus) in the Quantus repository and provides documentation
for how to {doc}`install </getting_started/installation>` Quantus, how to {doc}`contribute </docs_dev/CONTRIBUTING>` and details on the {doc}`API </docs_api/modules>`.
For further guidance on what to think about when applying Quantus, please read the {doc}`user guidelines </guidelines/guidelines_and_disclaimers>`. Do you want to get started? Please have a look at our simple {doc}`toy example </getting_started/getting_started_example>` with PyTorch using MNIST data.
For more examples, check the [tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus/tree/main/tutorials) folder.

Quantus can be installed from PyPI (this way assumes that you have either [PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org) installed on your machine):

```setup
pip install quantus
```

For a more in-depth guide on how to install Quantus, read more [here](https://quantus.readthedocs.io/en/latest/getting_started/installation.html). This includes instructions for how to install a desired deep learning framework such as PyTorch or tensorflow together with Quantus.

# Contents

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
:caption: API Reference
:maxdepth: 1

docs_api/quantus
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


## Citation


If you find this toolkit or its companion paper
[**Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations**](https://arxiv.org/abs/2202.06861)
interesting or useful in your research, please use the following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2023quantus,
  author  = {Anna Hedstr{\"{o}}m and Leander Weber and Daniel Krakowczyk and Dilyara Bareeva and Franz Motzkus and Wojciech Samek and Sebastian Lapuschkin and Marina Marina M.{-}C. H{\"{o}}hne},
  title   = {Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {34},
  pages   = {1--11},
  url     = {http://jmlr.org/papers/v24/22-0142.html}
}
```

When applying the individual metrics of Quantus, please make sure to also properly cite the work of the original authors.
You can find the relevant citations in the documentation of each respective metric {doc}`here </docs_api/modules>`.
