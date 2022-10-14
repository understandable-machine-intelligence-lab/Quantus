<p align="center">
  <img width="350" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/quantus_logo.png">
</p>
<!--<h1 align="center"><b>Quantus</b></h1>-->
<h3 align="center"><b>A toolkit to evaluate neural network explanations</b></h3>
<p align="center">
  PyTorch and Tensorflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/tutorial_basic_example_all_metrics.ipynb)
[![Python package](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml)
[![Code coverage](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/codecov.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/codecov.yml)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI version](https://badge.fury.io/py/quantus.svg)](https://badge.fury.io/py/quantus)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/quantus/badge/?version=latest)](https://quantus.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/github/understandable-machine-intelligence-lab/Quantus/coverage.svg?branch=master)](https://codecov.io/github/understandable-machine-intelligence-lab/Quantus?branch=master)

_Quantus is currently under active development so carefully note the Quantus release version to ensure reproducibility of your work._

[📑 Shortcut to paper!](https://arxiv.org/abs/2202.06861)
        
## News and Highlights! :rocket:

- Please see our [latest release](https://github.com/understandable-machine-intelligence-lab/Quantus/releases) which minor version includes some [heavy API changes](https://github.com/understandable-machine-intelligence-lab/Quantus/releases/tag/v0.2.0)!
- Offers more than **30+ metrics in 6 categories** for XAI evaluation 
- Supports different data types (image, time-series, NLP next up!) and models (PyTorch and Tensorflow)
- Latest metrics additions:
    - <b>Infidelity </b><a href="https://arxiv.org/abs/1901.09392">(Chih-Kuan, Yeh, et al., 2019)</a>
    - <b>ROAD </b><a href="https://arxiv.org/abs/2202.00449">(Rong, Leemann, et al., 2022)</a>
    - <b>Focus </b><a href="https://arxiv.org/abs/2109.15035">(Arias et al., 2022)</a>
    - <b>Consistency </b><a href="https://arxiv.org/abs/2202.00734">(Dasgupta et al., 2022)</a>
    - <b>Sufficiency </b><a href="https://arxiv.org/abs/2202.00734">(Dasgupta et al., 2022)</a>

## Citation

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

When applying the individual metrics of Quantus, please make sure to also properly cite the work of the original authors (as linked above).

## Table of contents

* [Library overview](#library-overview)
* [Installation](#installation)
* [Getting started](#getting-started)
* [Tutorials](#tutorials)
* [Misc functionality](#miscellaneous-functionality)
* [Contributing](#contributing)
<!--* [Citation](#citation)-->

## Library overview 

Simple visual comparison of eXplainable Artificial Intelligence (XAI) methods is often not sufficient to decide which explanation method works best as shown exemplary in Figure a) for four gradient-based methods — Saliency (Mørch et al., 1995; Baehrens et al., 2010), Integrated Gradients (Sundararajan et al., 2017), GradientShap (Lundberg and Lee, 2017) or FusionGrad (Bykov et al., 2021), yet it is a common practice for evaluation XAI methods in absence of ground truth data.

Therefore, we developed Quantus, an easy to-use yet comprehensive toolbox for quantitative evaluation of explanations — including 30+ different metrics. 
With Quantus, we can obtain richer insights on how the methods compare e.g., b) by holistic quantification on several evaluation criteria and c) by providing sensitivity analysis of how a single parameter e.g. the pixel replacement strategy of a faithfulness test influences the ranking of the XAI methods.

</p>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/viz.png">
</p>
 
This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of XAI research — to help automate the task of _XAI quantification_. Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation or 6) axiomatic metrics (note, however, that the categories are oftentimes mentioned under different naming conventions e.g., 'robustness' is often replaced for 'stability' or 'sensitivity' and 'faithfulness' is commonly interchanged for 'fidelity'). The library contains implementations of the following evaluation metrics:

<details>
  <summary><b>Faithfulness</b></summary>
quantifies to what extent explanations follow the predictive behaviour of the model (asserting that more important features play a larger role in model outcomes)
 <br><br>
  <ul>
    <li><b>Faithfulness Correlation </b><a href="https://www.ijcai.org/Proceedings/2020/0417.pdf">(Bhatt et al., 2020)</a>: iteratively replaces a random subset of given attributions with a baseline value and then measuring the correlation between the sum of this attribution subset and the difference in function output 
    <li><b>Faithfulness Estimate </b><a href="https://arxiv.org/pdf/1806.07538.pdf">(Alvarez-Melis et al., 2018)</a>: computes the correlation between probability drops and attribution scores on various points
    <li><b>Monotonicity Metric </b><a href="https://arxiv.org/abs/1909.03012">(Arya et al. 2019)</a>: starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuring the effect on model performance
    <li><b>Monotonicity Metric </b><a href="https://arxiv.org/pdf/2007.07584.pdf"> (Nguyen et al, 2020)</a>: measures the spearman rank correlation between the absolute values of the attribution and the uncertainty in the probability estimation
    <li><b>Pixel Flipping </b><a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140">(Bach et al., 2015)</a>: captures the impact of perturbing pixels in descending order according to the attributed value on the classification score
    <li><b>Region Perturbation </b><a href="https://arxiv.org/pdf/1509.06321.pdf">(Samek et al., 2015)</a>: is an extension of Pixel-Flipping to flip an area rather than a single pixel
    <li><b>Selectivity </b><a href="https://arxiv.org/pdf/1706.07979.pdf">(Montavon et al., 2018)</a>: measures how quickly an evaluated prediction function starts to drop when removing features with the highest attributed values
    <li><b>SensitivityN </b><a href="https://arxiv.org/pdf/1711.06104.pdf">(Ancona et al., 2019)</a>: computes the correlation between the sum of the attributions and the variation in the target output while varying the fraction of the total number of features, averaged over several test samples
    <li><b>IROF </b><a href="https://arxiv.org/pdf/2003.08747.pdf">(Rieger at el., 2020)</a>: computes the area over the curve per class for sorted mean importances of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected), averaged over several test samples
    <li><b>Infidelity </b><a href="https://arxiv.org/pdf/1901.09392.pdf">(Chih-Kuan, Yeh, et al., 2019)</a>: represents the expected mean square error between 1) a dot product of an attribution and input perturbation and 2) difference in model output after significant perturbation 
    <li><b>ROAD </b><a href="https://arxiv.org/pdf/2202.00449.pdf">(Rong, Leemann, et al., 2022)</a>: measures the accuracy of the model on the test set in an iterative process of removing k most important pixels, at each step k most relevant pixels (MoRF order) are replaced with noisy linear imputations
    <li><b>Sufficiency </b><a href="https://arxiv.org/abs/2202.00734">(Dasgupta et al., 2022)</a>: measures the extent to which similar explanations have the same prediction label
</ul>
</details>

<details>
<summary><b>Robustness</b></summary>
measures to what extent explanations are stable when subject to slight perturbations of the input, assuming that model output approximately stayed the same
     <br><br>
<ul>
    <li><b>Local Lipschitz Estimate </b><a href="https://arxiv.org/pdf/1806.08049.pdf">(Alvarez-Melis et al., 2018)</a>: tests the consistency in the explanation between adjacent examples
    <li><b>Max-Sensitivity </b><a href="https://arxiv.org/pdf/1901.09392.pdf">(Yeh et al., 2019)</a>: measures the maximum sensitivity of an explanation using a Monte Carlo sampling-based approximation
    <li><b>Avg-Sensitivity </b><a href="https://arxiv.org/pdf/1901.09392.pdf">(Yeh et al., 2019)</a>: measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation
    <li><b>Continuity </b><a href="https://arxiv.org/pdf/1706.07979.pdf">(Montavon et al., 2018)</a>: captures the strongest variation in explanation of an input and its perturbed version
    <li><b>Consistency </b><a href="https://arxiv.org/abs/2202.00734">(Dasgupta et al., 2022)</a>: measures the probability that the inputs with the same explanation have the same prediction label
</ul>
</details>

<details>
<summary><b>Localisation</b></summary>
tests if the explainable evidence is centered around a region of interest (RoI) which may be defined around an object by a bounding box, a segmentation mask or, a cell within a grid
     <br><br>
<ul>
    <li><b>Pointing Game </b><a href="https://arxiv.org/abs/1608.00507">(Zhang et al., 2018)</a>: checks whether attribution with the highest score is located within the targeted object
    <li><b>Attribution Localization </b><a href="https://arxiv.org/abs/1910.09840">(Kohlbrenner et al., 2020)</a>: measures the ratio of positive attributions within the targeted object towards the total positive attributions
    <li><b>Top-K Intersection </b><a href="https://arxiv.org/abs/2104.14995">(Theiner et al., 2021)</a>: computes the intersection between a ground truth mask and the binarized explanation at the top k feature locations
    <li><b>Relevance Rank Accuracy </b><a href="https://arxiv.org/abs/2003.07258">(Arras et al., 2021)</a>: measures the ratio of highly attributed pixels within a ground-truth mask towards the size of the ground truth mask
    <li><b>Relevance Mass Accuracy </b><a href="https://arxiv.org/abs/2003.07258">(Arras et al., 2021)</a>: measures the ratio of positively attributed attributions inside the ground-truth mask towards the overall positive attributions
    <li><b>AUC </b><a href="https://doi.org/10.1016/j.patrec.2005.10.010">(Fawcett et al., 2006)</a>: compares the ranking between attributions and a given ground-truth mask
    <li><b>Focus </b><a href="https://arxiv.org/abs/2109.15035">(Arias et al., 2022)</a>: quantifies the precision of the explanation by creating mosaics of data instances from different classes
</ul>
</details>

<details>
<summary><b>Complexity</b></summary>
captures to what extent explanations are concise i.e., that few features are used to explain a model prediction
     <br><br>
<ul>
    <li><b>Sparseness </b><a href="https://arxiv.org/abs/1810.06583">(Chalasani et al., 2020)</a>: uses the Gini Index for measuring, if only highly attributed features are truly predictive of the model output
    <li><b>Complexity </b><a href="https://arxiv.org/abs/2005.00631">(Bhatt et al., 2020)</a>: computes the entropy of the fractional contribution of all features to the total magnitude of the attribution individually
    <li><b>Effective Complexity </b><a href="https://arxiv.org/abs/2007.07584">(Nguyen at el., 2020)</a>: measures how many attributions in absolute values are exceeding a certain threshold
</ul>
</details>

<details>
<summary><b>Randomisation</b></summary>
tests to what extent explanations deteriorate as inputs to the evaluation problem e.g., model parameters are increasingly randomised
     <br><br>
<ul>
    <li><b>Model Parameter Randomisation </b><a href="https://arxiv.org/abs/1810.03292">(Adebayo et. al., 2018)</a>: randomises the parameters of single model layers in a cascading or independent way and measures the distance of the respective explanation to the original explanation
    <li><b>Random Logit Test </b><a href="https://arxiv.org/abs/1912.09818">(Sixt et al., 2020)</a>: computes for the distance between the original explanation and the explanation for a random other class
</ul>
</details>

<details>
<summary><b>Axiomatic</b></summary>
  assesses if explanations fulfill certain axiomatic properties
     <br><br>
<ul>
    <li><b>Completeness </b><a href="https://arxiv.org/abs/1703.01365">(Sundararajan et al., 2017)</a>: evaluates whether the sum of attributions is equal to the difference between the function values at the input x and baseline x'.
    <li><b>Non-Sensitivity </b><a href="https://arxiv.org/abs/2007.07584">(Nguyen at el., 2020)</a>: measures whether the total attribution is proportional to the explainable evidence at the model output (and referred to as Summation to Delta (Shrikumar et al., 2017), Sensitivity-n (slight variation, Ancona et al., 2018) and Conservation (Montavon et al., 2018))
    <li><b>Input Invariance </b><a href="https://arxiv.org/abs/1711.00867">(Kindermans et al., 2017)</a>: adds a shift to input, asking that attributions should not change in response (assuming the model does not)
</ul>
</details>

Additional metrics will be included in future releases.

**Disclaimers.** It is worth noting that the implementations of the metrics in this library have not been verified by the original authors. Thus any metric implementation in this library may differ from the original authors. Further, bear in mind that evaluation metrics for XAI methods are often empirical interpretations (or translations) of qualities that some researcher(s) claimed were important for explanations to fulfill, so it may be a discrepancy between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity. 

The first iteration has been developed primarily for image classification tasks, with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be both applications and explanation methods e.g., example-based methods where this library won't be applicable. Similarly, there is a couple of metrics that are popular but are considered out of scope for the first iteration of the library e.g., metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021).

Please read the user guidelines for further guidance on how to best use the library. 

## Installation

### Installing from PyPI

If you already have [PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org) installed on your machine, 
the most light-weight version of Quantus can be obtained from [PyPI](https://pypi.org/project/quantus/) as follows
(i.e., this means that additional explainability functionality, as well as ML frameworks will not be included):

```setup
pip install quantus
```

Alternatively, you can simply add the desired framework (in brackets), and it will be installed in addition to Quantus:
For PyTorch:

```setup
pip install "quantus[torch]"
```

For Tensorflow:

```setup
pip install "quantus[tensorflow]"
```

### Installing from requirements.txt

Alternatively, you can simply install from the requirements.txt found [here](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/requirements.txt),
however, this only installs with the default setup, requiring either PyTorch or Tensorflow:

```setup
pip install -r requirements.txt
```

### Installing XAI Library Support (PyPI only)

Most evaluation metrics in Quantus allow for a choice of either providing pre-computed explanations directly as an input,
or to instead make use of several wrappers implemented in `quantus.explain` around common explainability libraries.
The following XAI Libraries are currently supported:

**Captum**

To enable the use of wrappers around [Captum](https://captum.ai/), you can run:

```setup
pip install "quantus[captum]"
```

**tf-explain**

To enable the use of wrappers around [tf.explain](https://github.com/sicara/tf-explain), you can run:

```setup
pip install "quantus[tf-explain]"
```

**Zennit**

To use Quantus with support for the [Zennit](https://github.com/chr5tphr/zennit) library, you can run:

```setup
pip install "quantus[zennit]"
```

Note that the three options above will also install the respective required frameworks (i.e., PyTorch or Tensorflow),
if they are not already installed in your environment.

### Installing Tutorial Requirements

The Quantus tutorials have more requirements than the base package, which you can install by running

```setup
pip install "quantus[tutorials]"
```

### Full Installation

To simply install all of the above, you can run

```setup
pip install "quantus[full]"
```

### Package Requirements

The package requirments are as follows:
```
python>=3.7.0
pytorch>=1.10.1
tensorflow==2.6.2
tqdm==4.62.3
```

## Getting started

The following will give a short introduction for how to get started with Quantus.

**Note**: This example is based on the [PyTorch](https://pytorch.org/) framework, but we also support 
[Tensorflow](https://www.tensorflow.org), which would differ only in the {ref}`preliminaries <prelim>` 
(i.e., the model and data loading), 
as well as in the available XAI libraries.

### Preliminaries
(prelim)=
Quantus implements methods for the quantitative evaluation of XAI methods.
Generally, in order to apply these, you will need:
* A model (variable `model`)
* Input data and labels (variables `x_batch` and `y_batch`)
* Explanations to evaluate (variables `a_batch_*`)

#### Model and Data

Let's first load the model and the data. In this example, a pre-trained LeNet available from Quantus 
for the purpose of this tutorial is loaded, but generally you might use any Pytorch (or Tensorflow) model instead.

```python
import quantus
from quantus import LeNet
import torch
import torchvision

# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
model = LeNet()
model.load_state_dict(torch.load("tests/assets/mnist_model"))

# Load datasets and make loaders.
test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

# Load a batch of inputs and outputs to use for XAI evaluation.
x_batch, y_batch = iter(test_loader).next()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
```

#### Explanations

We still need some explanations to evaluate. 
For this, there are two possibilities in Quantus:

##### Using Pre-computed Explanations
Quantus allows you to evaluate explanations that you have already computed previously, 
assuming that they match the data you provide in `x_batch`. Let's say you have explanations 
for [Saliency](https://arxiv.org/abs/1312.6034) and [Integrated Gradients](https://arxiv.org/abs/1703.01365)
already pre-computed.

In that case, you can simply load these into corresponding variables `a_batch_saliency` 
and `a_batch_intgrad`:

```python
a_batch_saliency = load("path/to/precomputed/saliency/explanations")
a_batch_saliency = load("path/to/precomputed/intgrad/explanations")
```

##### Using Quantus XAI Wrappers
The second possibility (if you don't have the explanations you are interested in available already) 
is to simply obtain them from one of the many XAI frameworks out there, 
such as [Captum](https://captum.ai/), 
[Zennit](https://github.com/chr5tphr/zennit), 
[tf.explain](https://github.com/sicara/tf-explain),
or [iNNvestigate](https://github.com/albermax/innvestigate).

The following code example shows how to obtain explanations ([Saliency](https://arxiv.org/abs/1312.6034) 
and [Integrated Gradients](https://arxiv.org/abs/1703.01365), to be specific) 
using [Captum](https://captum.ai/):

```python
import captum
from captum.attr import Saliency, IntegratedGradients

# Generate Integrated Gradients attributions of the first batch of the test set.
a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy()
a_batch_intgrad = IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy()

# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

# Quick assert.
assert [isinstance(obj, np.ndarray) for obj in [x_batch, y_batch, a_batch_saliency, a_batch_intgrad]]

# You can use any function (not necessarily captum) to generate your explanations.
```

However, this can be tedious if you want to compare multiple explanation methods, 
or switch hyperparameters, or even the XAI library used to compute explanations.
For these reasons, the `quantus.explain` function offers wrappers around [Captum](https://captum.ai/), 
[Zennit](https://github.com/chr5tphr/zennit), and
[tf.explain](https://github.com/sicara/tf-explain), 
so that explanations do not need to be computed by hand as shown above, 
and complex evaluations can be performed using less code. 

The qualitative aspects of explanations 
may look fairly uninterpretable - since we lack ground truth of what the explanations
should be looking like, it is hard to draw conclusions about the explainable evidence 
that we see:

![drawing](../assets/mnist_example.png)

So, to quantitatively evaluate the explanations, we can apply Quantus. 

### Evaluating Explanations with Quantus

#### Quantus Metrics

Quantus implements XAI evaluation metrics from different categories 
(faithfulness, localisation, robustness, ...) which all inherit from the base `quantus.Metric` class. 
Metrics are designed as `Callables`. To apply a metric to your setting (e.g., [Max-Sensitivity](https://arxiv.org/abs/1901.09392)), 
they first need to be instantiated:

```python
max_sensitivity = quantus.MaxSensitivity()
```

and then applied to your model, data, and (pre-computed) explanations:

```python
result = max_sensitivity(
    model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch_salicency,
    device=device
)
```

Alternatively, if you want to employ the `quantus.explain` utility instead of pre-computing explanations,
you can call the metric like this:

```python
result = max_sensitivity(
    model=model,
    x_batch=x_batch,
    y_batch=y_batch,
    device=device,
    explain_func=quantus.explain,
    explain_func_kwargs={"method": "Saliency"})
)
```

#### Customising Metrics

The metrics for evaluating XAI methods are often quite sensitive to their respective hyperparameters. 
For instance, how explanations are normalised or whether signed or unsigned explanations are considered can have significant
impact on the results of the evaluation. However, some metrics require normalisation or unsigned values, while others are more flexible.

Therefore, different metrics can have different hyperparameters or default values in Quantus, which are documented in detail 
{doc}`here </docs_api/modules>`. We encourage users to read the respective documentation before applying each metric, 
to gain an understanding of the implications of altering each hyperparameter.

Nevertheless, for the purpose of robust evaluation, it makes sense to vary especially those hyperparameters that metrics tend to be
sensitive to. Generally, hyperparameters for each metric are separated as follows:

* Hyperparameters affecting the metric function itself are set in the `__init__` method of each metric. 
  Extending the above example of MaxSensitivity, various init hyperparameters can be set as follows:
    ```python
    max_sensitivity = quantus.MaxSensitivity(
        nr_samples=10,
        lower_bound=0.2,
        norm_numerator=quantus.fro_norm,
        norm_denominator=quantus.fro_norm,
        perturb_func=quantus.uniform_noise,
        similarity_func=quantus.difference
    )
    ```
* Hyperparameters affecting the inputs (data, model, explanations) to each metric are set in the `__call__` method of each metric.
  Extending the above example of MaxSensitivity, various call hyperparameters can be set as follows:
    ```python
    result = max_sensitivity(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        device=device,
        explain_func=quantus.explain,
        explain_func_kwargs={"method": "Saliency"},
        softmax=False
    )
    ```


#### Large-Scale Evaluations
Quantus also provides high-level functionality to support large-scale evaluations,
e.g., multiple XAI methods, multifaceted evaluation through several metrics, or a combination thereof.

To utilize `quantus.evaluate()`, you simply need to define two dictionaries:
* The **XAI Methods** you would like to evaluate:
    ```python
    xai_methods = {
        "Saliency": a_batch_saliency,
        "IntegratedGradients": a_batch_intgrad
    }
    ```
* The **Metrics** you would like to use for evaluation (each `__init__` parameter configuration counts as its own metric):
    ```python
    metrics = {
        "max-sensitivity-10": quantus.MaxSensitivity(nr_samples=10),
        "max-sensitivity-20": quantus.MaxSensitivity(nr_samples=20),
        "region-perturbation": quantus.RegionPerturbation(),
    }
    ```

After defining how to aggregate the measurements of each metric on each XAI-method, you can then simply run a large-scale evaluation as

```python
import numpy as np

agg_func = np.mean
metric_call_kwargs = {
  "model"=model,
  "x_batch"=x_batch,
  "y_batch"=y_batch,
  "softmax": False,
}

results = quantus.evaluate(
      metrics=metrics,
      xai_methods=xai_methods,
      agg_func=np.mean,
      **metric_call_kwargs
)
```

You can find a dedicated notebook similar to the example in this tutorial here: [
Getting started](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/Tutorial_Getting_Started.ipynb).

### Extending Quantus

With Quantus, one can flexibly extend the library's functionality, e.g., to adopt a customised explainer function
`explain_func` or to replace a function that perturbs the input `perturb_func` with a user-defined one.
If you are extending or replacing a function within the Quantus framework, make sure that your new function:

- has the same **return type**
- expects the same **arguments**

as the function you’re intending to replace.

Details on what datatypes and arguments that should be used for the different functions can be found in the respective 
function typing in {doc}`quantus.helpers</docs_api/quantus.helpers>`. 
For example, if you want to replace `similarity_func` in your evaluation, you can do as follows.

```python
import scipy
import numpy as np

def my_similarity_func(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate the similarity of a and b by subtraction."""
    return a - b

# Simply initalise the metric with your own function.
metric = quantus.LocalLipschitzEstimate(similarity_func=my_similar_func)
```

Similarly, if you are replacing or extending metrics, make sure they inherit from the `Metric` class in 
{doc}`quantus.metrics.base</docs_api/quantus.metrics.base>`. Each metric at least needs to implement the
`Metric.evaluate_instance` method.

## Miscellaneous

There are several miscellaneous helpers built-into Quantus intended for easier usability:

````python
# Interpret scores of a given metric.
metric_instance.interpret_scores

# Understand the hyperparameters of a metric.
sensitivity_scorer.get_params

# To list available metrics (and their corresponding categories).
quantus.AVAILABLE_METRICS

# To list available explainable methods.
quantus.AVAILABLE_XAI_METHODS

# To list available perturbation functions.
quantus.AVAILABLE_SIMILARITY_FUNCTIONS

# To list available similarity functions.
quantus.AVAILABLE_PERTURBATION_FUNCTIONS

# To list available normalisation function.
quantus.AVAILABLE_NORMALISATION_FUNCTIONS

# To get the scores of the last evaluated batch.
metric_instance_called.last_results
````

Per default, 
warnings are printed to shell with each metric initialisation in order to make the user attentive to the hyperparameters 
of the metric which may have great influence on the evaluation outcome. 
If you are running evaluation iteratively you might want to disable warnings, 
then set:

```disable_warnings = True```

in the params of the metric initalisation. Additionally, if you want to track progress while evaluating your explanations set:

```display_progressbar = True```

If you want to return an aggreagate score for your test samples you can set the following hyperparameter:

```return_aggregate = True```

for which you can specify an `aggregate_func` e.g., `np.mean` to use while aggregating the score for a given metric.

## Tutorials

Further tutorials are available that showcase the many types of analysis that can be done using Quantus.
For this purpose, please see notebooks in [tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/) folder which includes examples such as:
* [ImageNet Example All Metrics](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Example_All_Metrics.ipynb): shows how to instantiate the different metrics for ImageNet
* [Metric Parameterisation Analysis](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_Metric_Parameterisation_Analysis.ipynb): explores how sensitive a metric could be to its hyperparameters
* [Explanation Sensitivity Evaluation Model Training](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_Explanation_Sensitivity_Evaluation_Model_Training.ipynb): looks into how robustness of gradient-based explanations change as model gets increasingly accurate in its predictions
* [ImageNet Quantification with Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Quantification_with_Quantus.ipynb): benchmarks explanation methods under different types of analysis: qualitative, quantitative and sensitivity
... and more.

## Contributing

We welcome any sort of contribution to Quantus. For a detailed contribution guide, please refer to [Contributing](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/CONTRIBUTING.md) documentation first.
