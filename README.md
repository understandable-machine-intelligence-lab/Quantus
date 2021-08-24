<h1 align="center"><b>Quantus</b></h1>
<h3 align="center"><b>Quantus: a metrics toolbox to evaluate neural network explanations</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p> 
 
--------------

**A library that helps you understand to what extent your XAI explanations are doing anything useful.**   

<p align="center">
  <img src="samples/spider_image.png" alt="Visualization of how Quantus library can be used to highlight differences between explanation methods as well as the trade-offs between achieving a certain evaluation criteria." width="512"/>  
</p>

## Installation

To install requirements:

```setup
pip install -r requirements.txt
```

**Package requirements.**

```
Python >= 3.6.9
PyTorch >= 1.8
Captum == 0.4.0
```

## Library

This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of Explainable Artificial Intelligence (XAI) research. 
Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation/ unit or 6) axiomatic. 
As such, the `quantus library contains implementations of the following evaluation measures:

<span style="color:#ff0000">List TBC.</span>

* *Faithfulness:*
  * **[Faithfulness Correlation](https://www.ijcai.org/Proceedings/2020/0417.pdf) (Bhatt et al., 2020)**: insert description
  * **[Faithfulness Estimate](insert) (Alvarez-Melis et al., 2018a, 2018b)**: insert description
  * **Infidelity (Yeh at el., 2019)**:
  * **Monotonicity Metric (Nguyen at el., 2020)**: insert description
  * **Pixel Flipping (Bach et al., 2015)**: 
  * **Region Perturbation (Samek et al., 2015)**: 
  * **Selectivity (Montavan et al., 2018)**: 
  * **SensitivityN (Ancona et al., 2019)**:
* *Robustness:*
  * **Continuity (Montavon et al., 2018)**:
  * **Input Independence Rate (Yang et al., 2019)**: input independence measures the percentage of inputs where a functionally insignificant patch (e.g., a dog) does not affect explanations significantly
  * **Local Lipschitz Estimate (Alvarez-Melis et al., 2018a, 2018b)**:
  * **Max-Sensitivity (Yeh at el., 2019)**:
  * **Avg-Sensitivity (Yeh at el., 2019)**:
* *Localisation:*
  * **Pointing Game (Zhang et al., 2018)**:
  * **Attribution Localization (Kohlbrenner et al., 2020)**:
  * **TKI (Theiner et al., 2021)**:
  * **Relevance Rank Accuracy (Arras et al., 2021)**:
  * **Relevance Mass Accuracy (Arras et al., 2021)**:
* *Complexity:*
  * **Sparseness Test (Chalasani et al., 2020)**:
  * **Complexity Test (Bhatt et al., 2020)**:
* *Randomisation:*
  * **Model Parameter Randomisation Test**:
  * **Random Logit Test**:
* *Axiomatic:*
  * **Completeness (Sundararajan et al., 2017; **:
  * **Symmetry**:
  * **Sensitivity**:
  * **Dummy**:
  * **Input Invariance**:
    
**Note.** To avoid confusion, note that the six categories are often mentioned under different naming conventions e.g., "Robustness" is often replaced for "Stability" or "Sensitivity" and "Faithfulness" is commonly changed for "Fidelity".

**Scope.** There is a couple of metrics that are popular but have not been included in the first version of this library. Metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021) are considered out of scope of this project's first iteration.

This implementation is primarily motivated by image classification tasks. Further, it has been developed with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be explanation methods e.g., example-based methods where evaluation metrics in this library might not be suitable.   

## Getting started

To use the library, you'll need a couple of ingredients; a torch model, some input data and labels (to explain).

```python
import quantus
import torch
import torchvision

# Load a pre-trained classification model.
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Load datasets and make loaders.
test_set = torchvision.datasets.Caltech256(root='./sample_data', 
                                           download=True,
                                           transform=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                                                     torchvision.transforms.CenterCrop((224, 224)),
                                                                                     torchvision.transforms.ToTensor(),
                                                                                     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=12)

# Load a batch of inputs and outputs to use for evaluation.
x_batch, y_batch = iter(test_loader).next()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
```

Next, we generate some explanations for some test set samples that we wish to evaluate using `quantus` library.

```python
import captum
from captum.attr import Saliency, IntegratedGradients

a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
a_batch_intgrad = IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch, baselines=torch.zeros_like(inputs)).sum(axis=1)

# You can use any function (not necessarily captum) to generate your explanation. 
```
To evaluate explanations, there are two options. 

1) Either evaluate your explanations in a one-liner - by calling the instance of the metric class.

````python

metric_sensitivity = quantus.MaxSensitivity()
scores = metric_sensitivity(model=model,
                            x_batch=x_batch, 
                            y_batch=y_batch,
                            a_batch=a_batch_saliency,
                            **{"device": device, "img_size": 224, "normalize": True})
````

2) Or use `quantus.evaluate()` which is a high-level function that allow you to evaluate multiple XAI methods on several metrics at once.

```python
import numpy as np

metrics = {"Faithfulness correlation": quantus.FaithfulnessCorrelation(**{"subset_size": 32}),
           "max-Sensitivity": quantus.MaxSensitivity()}

xai_methods = {"Saliency": a_batch_saliency,
                "IntegratedGradients": a_batch_intgrad}

results = quantus.evaluate(evaluation_metrics=metrics,
                           explanation_methods=xai_methods, 
                           model=model,
                           x_batch=x_batch, 
                           y_batch=y_batch), 
                           agg_func=np.mean,
                           **{"device": device, "img_size": 224, "normalize": True})

# Summarise in a dataframe.      
df = pd.DataFrame(results)
df
```

Other miscellaneous functionality.

````python
# Interpret scores.
sensitivity_scorer.interpret_scores

# Understand what hyperparameters to tune.
sensitivity_scorer.list_hyperparameters

# To list available metrics
quantus.available_metrics
````

See more examples and use cases in the `/tutorials` folder.


* Compare explanation methods on different evalaution criteria (check out: `/tutorials/basic_example.ipynb`)
* Measure sensitivity of hyperparameter choice (check out: `/tutorials/hyperparameter_sensitivity.ipynb`)
* Understand how sensitivity of explanations change when a model is learning (check out: `/tutorials/training_robustness.ipynb`)
* Investigate to what extent metrics belonging to the same category score explanations similarliy (check out: `/tutorials/category_reliability.ipynb`)

... and more!

## User guideline

Always read the original publication to understand the context that the metric was introduced in (it may not be applicable to your specific task and/ or data domain). 
Spend time on understanding and investigate how the hyperparameters of the metrics influence the evaluation outcome; does changing the baseline value fundamentally change the evaluation outcome? 
Can we include a random explanation (as a control variant) to verify that indeed, the metric does reflect that such explanation is worse than other ones?

**Disclaimers.**

1. Implementation may differ from the original author(s)

Any metric implementation in this library may differ from the original authors. 
It is moreover likely that differences exist since 1) the source code of original publication is most often not made publicly available, 2) sometimes mathematical definitions are missing and 3) description of hyperparameter choice is left out.
   This leave room for (subjective) interpretation of the metric. 
The implementations of metrics in this library have not been verified by the original authors.

2. Discrepancy in operationalisation

Metrics for XAI methods are often empirical interpretations (or translations) of qualities that some researcher(s) claimed were important for explanations to fulfill. 
Hence it may be a discrepency between what the author claims to measure by the proposed metric and what is actually measured e.g., using entropy as an operationalisation of explanation complexity.   

3. Hyperparameters may (and should) change depending on application/ task and dataset/ domain

Metrics are often designed with a specific use case in mind and it is not always clear how to change the hyperparameters to make them suitable for another setting. 
Pay careful attention to how your hyperparameters should be tuned; what is a proper baseline value in your context i.e., that represents the notion of “missingness”?

## Feature list

For the next iteraiton of this library we will focus on:

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
	      title={Quantus: a metrics toolbox to evaluate neural network explanations}, 
	      author={},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}
