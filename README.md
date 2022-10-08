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

- Please see our latest release [v0.2.2](https://github.com/understandable-machine-intelligence-lab/Quantus/releases/tag/v0.2.2) which minor version includes heavy changes to the API. Read more [here](https://github.com/understandable-machine-intelligence-lab/Quantus/releases/tag/v0.2.0)!
- New updated [API documentation](https://quantus.readthedocs.io/en/latest/)
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

Quantus can be installed from [PyPI](https://pypi.org/project/quantus/) 
(this way assumes that you have either `torch` or `tensorflow` already installed on your machine).  

```setup
pip install quantus
```

If you don't have `torch` or `tensorflow` installed, you can simply add the package you want and install it simultaneously.

```setup
pip install "quantus[torch]"
```
Or, alternatively for `tensorflow` you run:

```setup
pip install "quantus[tensorflow]"
```

Additionally, if you want to use the basic explainability functionality such as `quantus.explain` in your evaluations, you can run `pip install "quantus[extras]"` (this step requires that either `torch` or `tensorflow` is installed). 
To use Quantus with `zennit` support, install in the following way: `pip install "quantus[zennit]"`.  

Alternatively, simply install requirements.txt (again, this requires that either `torch` or `tensorflow` is installed and won't include the explainability functionality to the installation):

```setup
pip install -r requirements.txt
```

### Package requirements

```
python>=3.7.0
pytorch>=1.10.1
tensorflow==2.6.2
tqdm==4.62.3
```

## Getting started

To use the library, you'll need a couple of ingredients; a model, some input data and labels (to be explained). In this example, we use `torch` but we also support evaluation of `tensorflow` models.

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

Next, we generate some explanations for some test set samples that we wish to evaluate using Quantus library.

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

# You can use any function e.g., quantus.explain (not necessarily captum) to generate your explanations.
```
<p align="center">
    <img src="tutorials/assets/mnist_model_example.png" alt="drawing" width="450"/>
</p>

The qualitative aspects of the Saliency and Integrated Gradients explanations may look fairly uninterpretable - since we lack ground truth of what the explanations should be looking like, it is hard to draw conclusions about the explainable evidence that we see. So, to quantitatively evaluate the explanation we can apply Quantus. For this purpose, we may be interested in measuring how sensitive the explanations are to very slight perturbations. To this end, we can e.g., apply max-sensitivity by Yeh et al., 2019 to evaluate our explanations. With Quantus, we created two options for evaluation.

1) Either evaluate the explanations in a one-liner - by calling the instance of the metric class.

```python
# Return max sensitivity scores in an one-liner - by calling the metric instance.
quantus.MaxSensitivity(
    nr_samples=10,
    lower_bound=0.2,
    norm_numerator=quantus.fro_norm,
    norm_denominator=quantus.fro_norm,
    perturb_func=quantus.uniform_noise,
    similarity_func=quantus.difference,
)(model=model, 
   x_batch=x_batch,
   y_batch=y_batch,
   a_batch=None,
   device=device,
   explain_func=quantus.explain, 
   explain_func_kwargs={"method": "Saliency"})

```

2) Or use `quantus.evaluate()` which is a high-level function that allow you to evaluate multiple XAI methods on several metrics at once.

```python
import numpy as np

metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval),
           }

xai_methods = {"Saliency": a_batch_saliency,
               "IntegratedGradients": a_batch_intgrad}

results = evaluate(
        metrics=metrics,
        xai_methods=xai_methods,
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        agg_func=np.mean,
        explain_func_kwargs={},
    )

# Summarise results in a dataframe.
df = pd.DataFrame(results)
df
```

When comparing the max-Sensitivity scores for the Saliency and Integrated Gradients explanations, we can conclude that in this experimental setting, Saliency can be considered less robust (scores 0.41 +-0.15std) compared to Integrated Gradients (scores 0.17 +-0.05std). To replicate this simple example please find a dedicated notebook: [Getting started](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/Tutorial_Getting_Started.ipynb).

## Tutorials

To get a more comprehensive view of the previous example, there is many types of analysis that can be done using Quantus. For example, we could use Quantus to verify to what extent the results - that Integrated Gradients "wins" over Saliency - are reproducible over different parameterisations of the metric e.g., by changing the amount of noise `lower_bound` or the number of samples to iterate over `nr_samples`. With Quantus, we could further analyse if Integrated Gradients offers an improvement over Saliency also in other evaluation criteria such as faithfulness, randomisation and localisation.

For more use cases, please see notebooks in [tutorials](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/) folder which includes examples such as:
* [ImageNet Example All Metrics](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Example_All_Metrics.ipynb): shows how to instantiate the different metrics for ImageNet
* [Metric Parameterisation Analysis](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_Metric_Parameterisation_Analysis.ipynb): explores how sensitive a metric could be to its hyperparameters
* [Explanation Sensitivity Evaluation Model Training](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_Explanation_Sensitivity_Evaluation_Model_Training.ipynb): looks into how robustness of gradient-based explanations change as model gets increasingly accurate in its predictions
* [ImageNet Quantification with Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_ImageNet_Quantification_with_Quantus.ipynb): benchmarks explanation methods under different types of analysis: qualitative, quantitative and sensitivity

... and more.


## Miscellaneous functionality

With Quantus, one can flexibly extend the library's functionality e.g., to adopt a customised explainer function `explain_func` or to replace a function that perturbs the input `perturb_func` with a user-defined one. 
If you are replacing a function within the Quantus framework, make sure that your new function:
- returns the same datatype (e.g., np.ndarray or float) and,
- employs the same arguments (e.g., img=x, a=a)
as the function you’re intending to replace.

Details on what datatypes and arguments that should be used for the different functions can be found in the respective function typing in`quantus/helpers`. For example, if you want to replace `similar_func` in your evaluation, you can do as follows.

````python
import scipy
import numpy as np

def correlation_spearman(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate Spearman rank of two images (or explanations)."""
    return scipy.stats.spearmanr(a, b)[0]

def my_similar_func(a: np.array, b: np.array, **kwargs) -> float:
    """Calculate the similarity of a and b by subtraction."""
    return a - b

# Simply initalise the metric with your own function.
metric = LocalLipschitzEstimate(similar_func=my_similar_func)
````

To evaluate multiple explanation methods over several metrics at once we user can leverage the `evaluate` method in Quantus. There are also other miscellaneous functionality built-into Quantus that might be helpful:

````python
# Interpret scores of a given metric.
metric_instance.interpret_scores

# Understand what hyperparameters of a metric to tune.
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
With each metric intialisation, warnings are printed to shell in order to make the user attentive to the hyperparameters of the metric which may have great influence on the evaluation outcome. If you are running evaluation iteratively you might want to disable warnings, then set: 
        
```disable_warnings = True```

in the params of the metric initalisation. Additionally, if you want to track progress while evaluating your explanations set:

```display_progressbar = True```

If you want to return an aggreagate score for your test samples you can set the following hyperparameter:

```return_aggregate = True```

for which you can specify an `aggregate_func` e.g., `np.mean` to use while aggregating the score for a given metric.

## Contributing

We welcome any sort of contribution to Quantus. For a detailed contribution guide, please refer to [Contributing](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/CONTRIBUTING.md) documentation first.
