<h1 align="center"><b>Quantus</b></h1>
<h3 align="center"><b>A metrics toolbox to evaluate neural network explanations</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p>

--------------
<img src="quantus.png" alt="drawing" width="200"/>

<!--**A library that helps you understand your XAI explanations..**-->

<p align="center">
  <img src="samples/spider_image.png" alt="Visualisation of how Quantus library can help highlight differences between explanation methods as well as implicit trade-offs between various evaluation criteria." width="512"/>
</p>

**Quantus is currently is currently under active development!**

## Library

This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of Explainable Artificial Intelligence (XAI) research.
Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation or 6) axiomatic metrics.
(Note that in literature, the categories are often mentioned under different naming conventions e.g., 'robustness' is often replaced for 'stability' or 'sensitivity' and "'faithfulness' is commonly interchanged for 'fidelity'.)

The `quantus` library contains implementations of the following evaluation metrics:

<span style="color:#ff0000">List TBC.</span>

* *Faithfulness:*
  * **[Faithfulness Correlation](https://www.ijcai.org/Proceedings/2020/0417.pdf) (Bhatt et al., 2020)**: insert description
  * **[Faithfulness Estimate](https://arxiv.org/abs/1806.07538) (Alvarez-Melis et al., 2018a, 2018b)**: insert description
  * **[Infidelity](https://arxiv.org/abs/1901.09392) (Yeh at el., 2019)**:
  * **[Monotonicity-Arya](https://arxiv.org/abs/1909.03012) (Arya at el., 2019)**: insert description
  * **[Monotonicity-Nguyen](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**: insert description
  * **[Pixel-Flipping](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140 (Bach et al., 2015)**:
  * **[Region Perturbation](https://arxiv.org/abs/1509.06321) (Samek et al., 2015)**:
  * **[Selectivity](https://arxiv.org/abs/1706.07979) (Montavan et al., 2018)**:
  * **[SensitivityN](https://arxiv.org/abs/1711.06104) (Ancona et al., 2019)**:
  * **[IterativeRemovalOfFeatures](https://arxiv.org/abs/2003.08747) (Rieger et al., 2019)**:
* *Robustness:*
  * **[Local Lipschitz Estimate](https://arxiv.org/abs/1806.07538) (Alvarez-Melis et al., 2018a, 2018b)**:
  * **[Max-Sensitivity](https://arxiv.org/abs/1901.09392) (Yeh at el., 2019)**:
  * **[Avg-Sensitivity](https://arxiv.org/abs/1901.09392) (Yeh at el., 2019)**:
  * **[Continuity](https://arxiv.org/abs/1706.07979) (Montavon et al., 2018)**:
  * **[Input Independence Rate](https://arxiv.org/abs/1907.09701) (Yang et al., 2019)**: measures the percentage of inputs where a functionally insignificant patch (e.g., a dog) does not affect explanations significantly
* *Localisation:*
  * **[Pointing Game](https://arxiv.org/abs/1608.00507) (Zhang et al., 2018)**:
  * **[Attribution Localization](https://arxiv.org/abs/1910.09840) (Kohlbrenner et al., 2020)**:
  * **[TKI](https://arxiv.org/abs/2104.14995) (Theiner et al., 2021)**:
  * **[Relevance Rank Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**:
  * **[Relevance Mass Accuracy](https://arxiv.org/abs/2003.07258) (Arras et al., 2021)**:
* *Complexity:*
  * **[Sparseness](https://arxiv.org/abs/1810.06583) (Chalasani et al., 2020)**:
  * **[Complexity](https://arxiv.org/abs/2005.00631) (Bhatt et al., 2020)**:
  * **[Effective Complexity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**:
* *Randomisation:*
  * **[Model Parameter Randomization](https://arxiv.org/abs/1810.03292) (Adebayo et al., 2018)**:
  * **[Random Logit Test](https://arxiv.org/abs/1912.09818) (Sixt et. al., 2020)**:
* *Axiomatic:*
  * **[Completeness](https://arxiv.org/abs/1703.01365) (Sundararajan et al., 2017) (and also, Summation to Delta (Shrikumar et al., 2017) Sensitivity-n (slight variation, Ancona et al., 2018) Conservation (Montavon et al., 2018))**:
  * **[Non-Sensitivity](https://arxiv.org/abs/2007.07584) (Nguyen at el., 2020)**:
  <!--* **Symmetry**:-->
  <!--* **Dummy**:-->
  <!--* **Input Invariance**:-->

**Scope.** There is a couple of metrics that are popular but have not been included in the first version of the library.
Metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021) are considered out of scope of the first iteration.

It is worth nothing that this implementation primarily is motivated by image classification tasks. Further, it has been developed with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be both applications and explanation methods e.g., example-based methods where this library won't be useful.


## Installation

To install requirements:

```setup
pip install -r requirements.txt
```

Package requirements.

```
Python >= 3.6.9
PyTorch >= 1.8
Captum == 0.4.0
```


## Getting started

To use the library, you'll need a couple of ingredients; a torch model, some input data and labels (to be explained).

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

# You can use any function e.g., quantus.explain (not necessarily captum) to generate your explanations.
```
To evaluate explanations, there are two options.

1) Either evaluate your explanations in a one-liner - by calling the instance of the metric class.

````python

metric_sensitivity = quantus.MaxSensitivity()
scores = metric_sensitivity(model=model,
                            x_batch=x_batch,
                            y_batch=y_batch,
                            a_batch=a_batch_saliency,
                            **{"explain_func": quantus.explain, "device": device, "img_size": 224, "normalize": True})
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

Other miscellaneous functionality of `quantus` library.

````python
# Interpret scores.
sensitivity_scorer.interpret_scores

# Understand what hyperparameters to tune.
sensitivity_scorer.list_hyperparameters

# To list available metrics
quantus.available_metrics
````

See more examples and use cases in the `/tutorials` folder. For example,

* Compare explanation methods on different evaluation criteria (check out: `/tutorials/basic_example.ipynb`)
* Measure sensitivity of hyperparameter choice (check out: `/tutorials/hyperparameter_sensitivity.ipynb`)
* Understand how sensitivity of explanations change when a model is learning (check out: `/tutorials/training_robustness.ipynb`)
* Investigate to what extent metrics belonging to the same category score explanations similarly (check out: `/tutorials/category_reliability.ipynb`)

... and more!

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

<!--

## Cite this paper

To cite this paper use following Bibtex annotation:

	@misc{quantus,
	      title={Quantus: a metrics toolbox to evaluate neural network explanations},
	      author={},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}
-->


### Citations of metrics

