<p align="center">
  <img width="350" height="200" src="https://raw.githubusercontent.com/understandable-machine-intelligence-lab/Quantus/main/quantus_logo.png">
</p>
<!--<h1 align="center"><b>Quantus</b></h1>-->
<h3 align="center"><b>A metrics toolkit to evaluate neural network explanations</b></h3>
<p align="center">
  <i>PyTorch implementation</i>
</p>

[![Python package](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/python-package.yml)
[![Code coverage](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/codecov.yml/badge.svg)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions/workflows/codecov.yml)
![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![PyPI version](https://badge.fury.io/py/quantus.svg)](https://badge.fury.io/py/quantus)

<!--[![Build Status](https://github.com/understandable-machine-intelligence-lab/Quantus/workflows/CI/badge.svg?branch=master)](https://github.com/understandable-machine-intelligence-lab/Quantus/actions?query=workflow%3A%22CI%22)-->
<!--[![Documentation Status](https://readthedocs.org/projects/alibi/badge/?version=latest)](https://docs.seldon.io/projects/qyabtys/en/latest/?badge=latest)-->
<!--[GitHub Licence](https://img.quantus.io/github/license/understandable-machine-intelligence-lab/Quantus.svg)-->
<!--[![Slack channel](https://img.qauntus.io/badge/chat-on%20slack-e51670.svg)](https://join.slack.com/t/seldondev/shared_invite/zt-vejg6ttd-ksZiQs3O_HOtPQsen_labg)-->

<!--**A library that helps you understand your XAI explanations..**-->
<!--
<p align="center">
  <img src="samples/spider_image.png" alt="Visualisation of how Quantus library can help highlight differences between explanation methods as well as implicit trade-offs between various evaluation criteria." width="512"/>
</p>
-->
_Quantus is currently under active development!_


<!--### Citation

If you find this library helpful in speeding up your research please cite using the following Bibtex annotation:

@misc{quantus,
      title={Quantus: Github repository},
      author={Anna Hedström, Leander Weber, Wojciech Samek, Sebastian Lapuschkin, Marina Höhne},
      year={2021},
      eprint={2106.10185},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}
-->

## Library content

This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of Explainable Artificial Intelligence (XAI) research. Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation or 6) axiomatic metrics. It is important to note here that in XAI literature, the categories are often mentioned under different naming conventions e.g., 'robustness' is often replaced for 'stability' or 'sensitivity' and "'faithfulness' is commonly interchanged for 'fidelity'.)

The library contains implementations of the following evaluation metrics:

* **Faithfulness:** quantifies to what extent explanations follow the predictive behaviour of the model (asserting that more important features play a larger role in model outcomes)
  * **[Faithfulness Correlation](https://www.ijcai.org/Proceedings/2020/0417.pdf) (Bhatt et al., 2020)**: iteratively replaces a random subset of given attributions with a baseline value and then measuring the correlation between the sum of this attribution subset and the difference in function output
  * **[Faithfulness Estimate](https://arxiv.org/pdf/1806.07538.pdf) (Alvarez-Melis et al., 2018a, 2018b)**: computes the correlation between probability drops and attribution scores on various points
  <!--* **[Infidelity](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf) (Yeh at el., 2019)**: represents the expected mean-squared error between the explanation multiplied by a meaningful input perturbation and the differences between the predictor function at its input and perturbed input-->
  * **[Monotonicity Metric Arya](https://arxiv.org/abs/1909.03012) (Arya et al., 2019)**: starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuing the effect on model performance
  * **[Monotonicity Metric Nguyen](https://arxiv.org/pdf/2007.07584.pdf) (Nguyen et al., 2020)**: measures the spearman rank correlation between the absolute values of the attribution and the uncertainty in the probability estimation
  * **[Pixel Flipping](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (Bach et al., 2015)**: captures the impact of perturbing pixels in descending order according to the attributed value on the classification score
  * **[Region Perturbation](https://arxiv.org/pdf/1509.06321.pdf) (Samek et al., 2015)**: is an extension of Pixel-Flipping to flip an area rather than a single pixel
  * **[Selectivity](https://arxiv.org/pdf/1706.07979.pdf) (Montavon et al., 2018)**: measures how quickly an evaluated prediction function starts to drop when removing features with the highest attributed values
  * **[SensitivityN](https://arxiv.org/pdf/1711.06104.pdf) (Ancona et al., 2019)**: computes the corerlation between the sum of the attributions and the variation in the target output while varying the fraction of the total number of features, averaged over several test samples
  * **[IROF](https://arxiv.org/pdf/2003.08747.pdf) (Iterative Removal of Features) (Rieger et al., 2020)**: computes the area over the curve per class for sorted mean importances of feature segments (superpixels) as they are iteratively removed (and prediction scores are collected), averaged over several test samples
* **Robustness:** measures to what extent explanations are stable when subject to slight perturbations of the input, assuming that model output approximately stayed the same
  * **[Local Lipschitz Estimate](https://arxiv.org/pdf/1806.08049.pdf) (Alvarez-Melis et al., 2018a, 2018b)**: tests the consistency in the explanation between adjacent examples
  * **[Max-Sensitivity](https://arxiv.org/pdf/1901.09392.pdf) (Yeh et al., 2019)**: measures the maximum sensitivity of an explanation using a Monte Carlo sampling-based approximation
  * **[Avg-Sensitivity](https://arxiv.org/pdf/1901.09392.pdf) (Yeh et al., 2019)**: measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation
  * **[Continuity](https://arxiv.org/pdf/1706.07979.pdf) (Montavon et al., 2018)**: captures the strongest variation in explanation of an input and it's perturbed version
  <!--* **[Input Independence Rate](https://arxiv.org/pdf/1907.09701.pdf) (Yang et al., 2019)**: measures the percentage of inputs where a functionally insignificant patch (e.g., a dog) does not affect explanations significantly-->
* **Localisation:** tests if the explainable evidence is centered around the object of interest (as defined by a bounding box or similar segmentation mask)
  * **[Pointing Game](https://arxiv.org/abs/1608.00507) (Zhang et al., 2018)**: checks whether attribution with the highest score is located within the targeted object
  * **[Attribution Localization](https://arxiv.org/abs/1910.09840) (Kohlbrenner et al., 2020)**: measures the ratio of positive attributions within the targeted object towards the total positive attributions
  * **[Top-K Intersection](https://arxiv.org/abs/2104.14995) (Theiner et al., 2021)**: computes the intersection between a ground truth mask and the binarized explanation at the top k feature locations
  * **[Relevance Rank Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**: measures the ratio of highly attributed pixels within a ground-truth mask towards the size of the ground truth mask
  * **[Relevance Mass Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**: measures the ratio of positively attributed attributions inside the ground-truth mask towards the overall positive attributions
  * **[AUC](https://doi.org/10.1016/j.patrec.2005.10.010) (Fawcett et al., 2006)**: compares the ranking between attributions and a given ground-truth mask
* **Complexity:** captures to what extent explanations are concise i.e., that few features are used to explain a model prediction
  * **[Sparseness](https://arxiv.org/abs/1810.06583) (Chalasani et al., 2020)**: uses the Gini Index for measuring, if only highly attributed features are truly predictive of the model output
  * **[Complexity](https://arxiv.org/abs/2005.00631) (Bhatt et al., 2020)**: computes the entropy of the fractional contribution of all features to the total magnitude of the attribution individually
  * **[Effective Complexity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**: measures how many attributions in absolute values are exceeding a certain threshold
* **Randomisation:** tests to what extent explanations deteriorate as model parameters are increasingly randomised
  * **[Model Parameter Randomisation](https://arxiv.org/abs/1810.03292) (Adebayo et al., 2018)**: randomises the parameters of single model layers in a cascading or independent way and measures the distance of the respective explanation to the original explanation
  * **[Random Logit Test](https://arxiv.org/abs/1912.09818) (Sixt et. al., 2020)**: computes for the distance between the original explanation and the explanation for a random other class
* **Axiomatic:** assesses if explanations fulfill certain axiomatic properties
  * **[Completeness](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017) (and referred to as Summation to Delta (Shrikumar et al., 2017) Sensitivity-n (slight variation, Ancona et al., 2018) Conservation (Montavon et al., 2018))**: measures whether the total attribution is proportional to the explainable evidence at the model output
  * **[Non-Sensitivity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020) (and referred to as Null Player, Ancona et al., 2019), Dummy (Montavon et al., 2018))**: measures if zero-importance is only assigned to features that the model is not functionally dependent on
  * **[Input Invariance](https://arxiv.org/abs/1711.00867)** (Kindermans et al., 2017): adds a shift to input, asking that attributions should not change in response (assuming the model does not)
  <!--* **Symmetry**:-->
  

Additional metrics will be included in future releases.

**Scope.** There is a couple of metrics that are popular but have not been included in the first version of this library.
Metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021) are considered out of scope of the first iteration.

**Motivation.** It is worth nothing that this implementation primarily is motivated by image classification tasks. Further, it has been developed with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be both applications and explanation methods e.g., example-based methods where this library won't be applicable.

**Disclaimers.** Note that the implementations of metrics in this library have not been verified by the original authors. Thus any metric implementation in this library may differ from the original authors. Also, metrics for XAI methods are often empirical interpretations (or translations) of qualities that some researcher(s) claimed were important for explanations to fulfill. Hence it may be a discrepancy between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity. Please read the user guidelines for further guidance on how to best use the library.

## Installation

<!--Quantus can be installed from [PyPI](https://pypi.org/project/quantus/0.0.1/):

```setup
pip install quantus
````
-->
To install requirements:

```setup
pip install -r requirements.txt
```

Package requirements:

```
Python >= 3.6.9
PyTorch >= 1.8
Captum >= 0.4.0
```

## Getting started

To use the library, you'll need a couple of ingredients; a torch model, some input data and labels (to be explained).

```python
import quantus
import torch
import torchvision

# Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
model = LeNet()
model.load_state_dict(torch.load("tutorials/assets/mnist"))

# Load datasets and make loaders.
test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

# Load a batch of inputs and outputs to use for XAI evaluation.
x_batch, y_batch = iter(test_loader).next()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Next, we generate some explanations for some test set samples that we wish to evaluate using quantus library.

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
    <img src="tutorials/assets/mnist_example.png" alt="drawing" width="450"/>
</p>

The qualitative aspects of the Saliency and Integrated Gradients explanations may look fairly uninterpretable - since we lack ground truth of what the explanations should be looking like, it is hard to draw conclusions about the explainable evidence that we see. So, to quantitatively evaluate the explanation we can apply Quantus. For this purpose, we may be interested in measuring how sensitive the explanations are to very slight perturbations. To this end, we can e.g., apply max-sensitivity by Yeh et al., 2019 to evaluate our explanations. With Quantus, we created two options for evaluation.

1) Either evaluate the explanations in a one-liner - by calling the instance of the metric class.

```python
# Define params for evaluation.
params_eval = {
  "nr_samples": 10,
  "perturb_radius": 0.1,
  "norm_numerator": quantus.fro_norm,
  "norm_denominator": quantus.fro_norm,
  "perturb_func": quantus.uniform_sampling,
  "similarity_func": quantus.difference,
  "img_size": 28, 
  "nr_channels": 1,
  "normalise": False, 
  "abs": False,
  "disable_warnings": True,
}

# Return max sensitivity scores in an one-liner - by calling the metric instance.
scores_saliency = quantus.MaxSensitivity(**params_eval)(model=model,
                                                        x_batch=x_batch,
                                                        y_batch=y_batch,
                                                        a_batch=a_batch_saliency,
                                                        **{"explain_func": quantus.explain, 
                                                           "method": "Saliency", 
                                                           "device": device})
```

2) Or use `quantus.evaluate()` which is a high-level function that allow you to evaluate multiple XAI methods on several metrics at once.

```python
import numpy as np

metrics = {"max-Sensitivity": quantus.MaxSensitivity(**params_eval),
           }

xai_methods = {"Saliency": a_batch_saliency,
               "IntegratedGradients": a_batch_intgrad}

results = quantus.evaluate(metrics=metrics,
                           xai_methods=xai_methods,
                           model=model,
                           x_batch=x_batch,
                           y_batch=y_batch,
                           agg_func=np.mean,
                           **{"explain_func": quantus.explain, "device": device})
# Summarise results in a dataframe.
df = pd.DataFrame(results)
df
```

When comparing the max-Sensitivity scores for the Saliency and Integrated Gradients explanations, we can conclude that in this experimental setting, Saliency can be considered less robust (scores 0.41 +-0.15std) compared to Integrated Gradients (scores 0.17 +-0.05std). To replicate this simple example please find a dedicated notebook: [Getting started](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/tutorial_getting_started.ipynb).

## Tutorials

To get a more comprehensive view of the previous example, there is many types of analysis that can be done using Quantus. For example, we could use Quantus to verify to what extent the results - that Integrated Gradients "wins" over Saliency - are reproducible over different parameterisations of the metric e.g., by changing the amount of noise `perturb_radius` or the number of samples to iterate over `nr_samples`. With Quantus, we could further analyse if Integrated Gradients offers an improvement over Saliency also in other evaluation criteria such as faithfulness, randomisation and localisation.

For more use cases, please see notebooks in `/tutorials` folder which includes examples such as

* [Basic example all metrics](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/tutorial_basic_example_all_metrics.ipynb): shows how to instantiate the different metrics for ImageNet
* [Metrics' parameterisation sensitivity](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/tutorial_sensitivity_parameterisation.ipynb): explores how sensitive a metric could be to its hyperparameters
* [Understand how explanations robustness develops during model training](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/tutorial_model_training_explanation_robustness.ipynb): looks into how robustness of gradient-based explanations change as model gets increasingly accurate in its predictions
<!--* Investigate to what extent metrics belonging to the same category score explanations similarly (check out: `/tutorials/category_reliability.ipynb`)-->

... and more.


## Misc functionality

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

To evaluate multiple explanation methods over several metrics at oncewe user can leverage the `evaluate` method in Quantus. There are also other miscellaneous functionality built-into Quantus that might be helpful:

````python
# Interpret scores.
quantus.evaluate

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

# To get the scores of all the evaluated batches.
metric_instance_called.all_results
````
With each metric intialisation, warnings are printed to shell in order to make the user attentive to the hyperparameters of the metric which may have great influence on the evaluation outcome. If you are running evaluation iteratively you might want to disable warnings, then set: 
        
```disable_warnings = True```

in the params of the metric initalisation.

## Contributing

If you would like to contribute to this project or add your metric to evaluate explanations please open an issue or submit a pull request.

#### Code Style
Code is written to follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) and for docstrings we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html).
We use [flake8](https://pypi.org/project/flake8/) for quick style checks and [black](https://github.com/psf/black) for code formatting with a line-width of 88 characters per line.

#### Testing
Tests are written using [pytest](https://github.com/pytest-dev/pytest) and executed together with [codecov](https://github.com/codecov/codecov-action) for coverage reports.


## Disclaimers

**1. Implementation may differ from the original author(s)**

Note that the implementations of metrics in this library have not been verified by the original authors. Thus any metric implementation in this library may differ from the original authors. It is moreover likely that differences exist since 1) the source code of original publication is most often not made publicly available, 2) sometimes the mathematical definition of the metric is missing and/ or 3) the description of hyperparameter choice was left out. This leaves room for (subjective) interpretations.

**2. Discrepancy in operationalisation is likely**

Metrics for XAI methods are often empirical interpretations (or translations) of qualities that researcher(s) stated were important for explanations to fulfil. Hence it may be a discrepancy between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity.

**3. Hyperparameters may (and should) change depending on application/ task and dataset/ domain**

Metrics are often designed with a specific use case in mind e.g., in an image classification setting. Thus it is not always clear how to change the hyperparameters to make them suitable for another setting. Pay careful attention to how your hyperparameters should be tuned; what is a proper baseline value in your context i.e., that represents the notion of “missingness”?

**4. Evaluation of explanations must be understood in its context; its application and of its kind**

 What evaluation metric to use is completely dependent on: 1) the type of explanation (explanation by example cannot be evaluated the same way as attribution-based/ feature-importance methods), 2) the application/ task: we may not require the explanations to fulfil certain criteria in some context compared to others e.g., multi-label vs single label classification 3) the dataset/ domain: text vs images e.g, different dependency structures between features exist, and preprocessing of the data, leading to differences on what the model may perceive, and how attribution methods can react to that (prime example: MNIST in range  [0,1] vs [-1,1] and any NN) and 4) the user (most evaluation metrics are founded from principles of what a user want from its explanation e.g., even in the seemingly objective measures we are enforcing our preferences e.g., in TCAV "explain in a language we can understand", object localisation "explain over objects we think are important", robustness "explain similarly over things we think looks similar" etc. Thus it is important to define what attribution quality means for each experimental setting.

**5. Evaluation (and explanations) will be unreliable if the model is not robust**

Evaluation will fail if you explain a poorly trained model. If the model is not robust, then explanations cannot be expected to be meaningful or interpretable [1, 2]. If the model achieves high predictive performance, but for the wrong reasons (e.g., Clever Hans, Backdoor issues) [3, 4], there is likely to be unexpected effects on the localisation metrics (which generally captures how well explanations are able to centre attributional evidence on the object of interest).

**6. Evaluation outcomes can be true to data or true to model**

Interpretation of evaluation outcome will differ depending on whether we prioritise that attributions are faithful to data or to the model [5, 6]. As explained in [5], imagine if a model is trained to use only one of two highly correlated features. The explanation might then rightly point out that this one feature is important (and that the other correlated feature is not). But if we were to re-train the model, the model might now pick the other feature as basis for prediction, for which the explanation will consequently tell another story --- that the other feature is important. Since the explanation function have returned conflicting information about what features are important --- we might now believe that the explanation function in itself is unstable. But this may not necessarily be true --- in this case, the explanation has remained faithful to the model but not the data. As such, in the context of evaluation, to avoid misinterpretation of results, it may therefore be important to articulate what you care most about explaining.

#### References

[1] P. Chalasani, J. Chen, A. R. Chowdhury, X. Wu, and S. Jha, “Concise explanations of neural  networks using adversarial training,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, pp. 1383–1391, 2020.

[2] N. Bansal, C. Agarwal, and A. Nguyen, “SAM: the sensitivity of attribution methods to  hyperparameters,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition,  CVPR Workshops 2020, Seattle, WA, USA, June 14-19, 2020. Computer Vision Foundation IEEE, pp. 11–21, 2020.

[3] S. Lapuschkin, S. Wäldchen, A. Binder, G. Montavon, W. Samek, and K.-R. Müller, “Unmasking clever hans predictors and assessing what machines really learn,” Nature Communications, vol. 10, p. 1096, 2019.

[4] C. J. Anders, L. Weber, D. Neumann, W. Samek, K.-R. Müller, and S. Lapuschkin, “Finding  and removing clever hans: Using explanation methods to debug and improve deep models,”  Information Fusion, vol. 77, pp. 261–295, 2022.

[5] P. Sturmfels, S. Lundberg, and S. Lee. "Visualizing the impact of feature attribution baselines." Distill 5, no. 1: e22, 2020.

[6] D. Janzing, L. Minorics, and P. Blöbaum. "Feature relevance quantification in explainable AI: A causal problem." In International Conference on Artificial Intelligence and Statistics, pp. 2907-2916. PMLR, 2020.


<!--

## Feature list

For the next iteration, focus will be on the following items.

* Tensorflow compatibility
* Build a 'Quantifier' class:
  * Handling of cache, writing to file, saving output, hdf5
  * Parallelization capability
* Post-processing
  * Providing plots
  * Populating table/ overview graphs of various scores
* Other functionality
  * Incorporate dataset wide measures e.g., like SpRAy compatibility
    Perturbation outlier test, or detecting out-of-distribution samples
  * Smarter segmentation of images to perform SDC and SSC


## Cite this paper

To cite this paper use following Bibtex annotation:

	@misc{quantus,
	      title={Quantus: A Comprehensive Toolbox for Responsible Evaluation of Neural Network Explanations},
	      author={},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}

### Citations of metrics

-->




