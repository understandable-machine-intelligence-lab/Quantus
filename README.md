<h1 align="center"><b>Quantus</b></h1>
<h3 align="center"><b>A metrics toolbox to evaluate neural network explanations</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p>

--------------
<!--<img src="quantus.png" alt="drawing" width="600"/>-->

<!--**A library that helps you understand your XAI explanations..**-->
<!--
<p align="center">
  <img src="samples/spider_image.png" alt="Visualisation of how Quantus library can help highlight differences between explanation methods as well as implicit trade-offs between various evaluation criteria." width="512"/>
</p>
-->
_Quantus is currently under active development!_

## Library content

This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of Explainable Artificial Intelligence (XAI) research. Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation or 6) axiomatic metrics. It is important to note here that in XAI literature, the categories are often mentioned under different naming conventions e.g., 'robustness' is often replaced for 'stability' or 'sensitivity' and "'faithfulness' is commonly interchanged for 'fidelity'.)

The library contains implementations of the following evaluation metrics:

* *Faithfulness:*
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
* *Robustness:*
  * **[Local Lipschitz Estimate](https://arxiv.org/pdf/1806.08049.pdf) (Alvarez-Melis et al., 2018a, 2018b)**: tests the consistency in the explanation between adjacent examples
  * **[Max-Sensitivity](https://arxiv.org/pdf/1901.09392.pdf) (Yeh et al., 2019)**: measures the maximum sensitivity of an explanation using a Monte Carlo sampling-based approximation
  * **[Avg-Sensitivity](https://arxiv.org/pdf/1901.09392.pdf) (Yeh et al., 2019)**: measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation
  * **[Continuity](https://arxiv.org/pdf/1706.07979.pdf) (Montavon et al., 2018)**: captures the strongest variation in explanation of an input and it's perturbed version
  <!--* **[Input Independence Rate](https://arxiv.org/pdf/1907.09701.pdf) (Yang et al., 2019)**: measures the percentage of inputs where a functionally insignificant patch (e.g., a dog) does not affect explanations significantly-->
* *Localisation:*
  * **[Pointing Game](https://arxiv.org/abs/1608.00507) (Zhang et al., 2018)**: checks whether attribution with the highest score is located within the targeted object
  * **[Attribution Localization](https://arxiv.org/abs/1910.09840) (Kohlbrenner et al., 2020)**: measures the ratio of positive attributions within the targeted object towards the total positive attributions
  * **[Top-K Intersection](https://arxiv.org/abs/2104.14995) (Theiner et al., 2021)**: computes the intersection between a ground truth mask and the binarized explanation at the top k feature locations
  * **[Relevance Rank Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**: measures the ratio of highly attributed pixels within a ground-truth mask towards the size of the ground truth mask
  * **[Relevance Mass Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**: measures the ratio of positively attributed attributions inside the ground-truth mask towards the overall positive attributions
  * **[AUC](https://doi.org/10.1016/j.patrec.2005.10.010) (Fawcett et al., 2006)**: compares the ranking between attributions and a given ground-truth mask
* *Complexity:*
  * **[Sparseness](https://arxiv.org/abs/1810.06583) (Chalasani et al., 2020)**: uses the Gini Index for measuring, if only highly attributed features are truly predictive of the model output
  * **[Complexity](https://arxiv.org/abs/2005.00631) (Bhatt et al., 2020)**: computes the entropy of the fractional contribution of all features to the total magnitude of the attribution individually
  * **[Effective Complexity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**: measures how many attributions in absolute values are exceeding a certain threshold
* *Randomisation:*
  * **[Model Parameter Randomisation](https://arxiv.org/abs/1810.03292) (Adebayo et al., 2018)**: randomises the parameters of single model layers in a cascading or independent way and measures the distance of the respective explanation to the original explanation
  * **[Random Logit Test](https://arxiv.org/abs/1912.09818) (Sixt et. al., 2020)**: computes for the distance between the original explanation and the explanation for a random other class
* *Axiomatic:*
  * **[Completeness](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017) (and also, Summation to Delta (Shrikumar et al., 2017) Sensitivity-n (slight variation, Ancona et al., 2018) Conservation (Montavon et al., 2018))**: measures whether the total attribution is proportional to the explainable evidence at the model output
  * **[Non-Sensitivity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**: measures, if zero-importance is only assigned to features, that the model is not functionally dependent on
  <!--* **Symmetry**:-->
  <!--* **Dummy**:-->
  <!--* **Input Invariance**:-->

Additional metrics will be included in future releases.

**Scope.** There is a couple of metrics that are popular but have not been included in the first version of this library.
Metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021) are considered out of scope of the first iteration.

It is worth nothing that this implementation primarily is motivated by image classification tasks. Further, it has been developed with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be both applications and explanation methods e.g., example-based methods where this library won't be applicable.

**Disclaimers.** Note that the implementations of metrics in this library have not been verified by the original authors. Thus any metric implementation in this library may differ from the original authors. Also, metrics for XAI methods are often empirical interpretations (or translations) of qualities that some researcher(s) claimed were important for explanations to fulfill. Hence it may be a discrepancy between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity. Please read the user guidelines for further guidance on how to best use the library.

## Installation

To install requirements:

```setup
pip install -r requirements.txt
```

Package requirements.

```
Python >= 3.6.9
PyTorch >= 1.8
Captum >= 0.9.0
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

# Load a batch of inputs and outputs to use for evaluation.
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

The qualitative aspects of the Saliency and Integrated Gradients explanations may look fairly uninterpretable - since we lack ground truth of what the explanations should be looking like, it is hard to draw conclusions about the explainable evidence that we see.

To quantitatively evaluate the explanation we can use apply Quantus. As a starter we may be interested in measuring how sensitive the explanations are to very slight perturbations. For this, we apply max-sensitivity by Yeh et al., 2019 to evaluate our explanations. With Quantus, there are two options.

1) Either evaluate the explanations in a one-liner - by calling the instance of the metric class.

```python
# Return max sensitivity scores in an one-liner - by calling the metric instance.
scores_saliency = quantus.MaxSensitivity(**{
    "nr_samples": 10,
    "perturb_radius": 0.1,
    "norm_numerator": quantus.fro_norm,
    "norm_denominator": quantus.fro_norm,
    "perturb_func": quantus.uniform_sampling,
    "similarity_func": quantus.difference,
})(model=model,
   x_batch=x_batch,
   y_batch=y_batch,
   a_batch=a_batch_saliency,
   **{"explain_func": quantus.explain, "method": "Saliency", "device": device,
   "img_size": 28, "normalise": False, "abs": False})
```

2) Or use `quantus.evaluate()` which is a high-level function that allow you to evaluate multiple XAI methods on several metrics at once.

```python
import numpy as np

metrics = {"max-Sensitivity": quantus.MaxSensitivity(**{"nr_samples": 10,
                                                        "perturb_radius": 0.1,
                                                        "norm_numerator": quantus.fro_norm,
                                                        "norm_denominator": quantus.fro_norm,
                                                        "perturb_func": quantus.uniform_sampling,
                                                        "similarity_func": quantus.difference})}

xai_methods = {"Saliency": a_batch_saliency,
               "IntegratedGradients": a_batch_intgrad}

results = quantus.evaluate(evaluation_metrics=metrics,
                           explanation_methods=xai_methods,
                           model=model,
                           x_batch=x_batch,
                           y_batch=y_batch,
                           agg_func=np.mean,
                           **{"explain_func": quantus.explain, "device": device,
                           "img_size": 28, "normalise": False, "abs": False})
# Summarise results in a dataframe.
df = pd.DataFrame(results)
df
```

When comparing the max-Sensitivity scores for the Saliency and Integrated Gradients explanations, we can conclude that in this experimental setting, Saliency can be considered less robust (scores 0.41 +-0.15std) compared to Integrated Gradients (scores 0.17 +-0.05std). To get a more comprehensive view, we would use Quantus to verify to what extent this results are reproducible over different parameterisations of the metric e.g., by changing the amount of noise `perturb_radius` or the number of samples to iterate over 'nr_samples'. With Quantus, we would further analyse if Integrated Gradients offers an improvement over Saliency also in other evaluation criteria such as faithfulness, randomisation and localisation. To replicate this simple example please find a dedicated notebook under `/tutorials/getting_started.ipynb`.

### Tutorials

For more examples, please see notebooks in `/tutorials` folder. We have included some preliminary use cases such as:

* [Basic example all metrics](https://github.com/understandable-machine-intelligence-lab/quantus/blob/main/tutorials/tutorial_basic_example_all_metrics.ipynb): shows how to instantiate the different metrics for ImageNet••
* Measure sensitivity of hyperparameter choice (`/tutorials/sensitivity_parameterisation.ipynb`)
* Understand how sensitivity of explanations change when a model is learning (`/tutorials/model_training_explanation_robustness.ipynb`)
<!--* Investigate to what extent metrics belonging to the same category score explanations similarly (check out: `/tutorials/category_reliability.ipynb`)-->

... and more.

In addition to the `evaluate` method that allows the user to evaluate multiple explanation methods over several metrics at once, there are also other miscellaneous functionality of Quantus library that might be helpful:

````python
# Interpret scores.
metric_instance.interpret_scores

# Understand what hyperparameters to tune.
sensitivity_scorer.get_params

# To list available metrics.
quantus.available_metrics
````

### Contributing

If you would like to contribute to this project or add your metric to evaluate explanations please open an issue or submit a pull request.

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




