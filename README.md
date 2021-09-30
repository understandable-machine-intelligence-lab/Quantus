<h1 align="center"><b>Quantus</b></h1>
<h3 align="center"><b>A metrics toolbox to evaluate neural network explanations</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p> 
 
--------------

<!--**A library that helps you understand to what extent your explanations.**-->   

<p align="center">
  <img src="samples/spider_image.png" alt="Visualisation of how Quantus library can help highlight differences between explanation methods as well as implicit trade-offs between various evaluation criteria." width="512"/>  
</p>

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

## Library

This project started with the goal of collecting existing evaluation metrics that have been introduced in the context of Explainable Artificial Intelligence (XAI) research. 
Along the way of implementation, it became clear that XAI metrics most often belong to one out of six categories i.e., 1) faithfulness, 2) robustness, 3) localisation 4) complexity 5) randomisation or 6) axiomatic metrics. 
(Note that in literature, the categories are often mentioned under different naming conventions e.g., 'robustness' is often replaced for 'stability' or 'sensitivity' and "'faithfulness' is commonly interchanged for 'fidelity'.)

The `quantus` library contains implementations of the following evaluation measures:

<span style="color:#ff0000">List TBC.</span>

* *Faithfulness:*
  * **[Faithfulness Correlation](https://www.ijcai.org/Proceedings/2020/0417.pdf) (Bhatt et al., 2020)**: insert description
  * **[Faithfulness Estimate](insert) (Alvarez-Melis et al., 2018a, 2018b)**: insert description
  * **[Infidelity](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf) (Yeh at el., 2019)**: The explanation infidelity represents the expected mean-squared error between the explanation multiplied by a meaningful input perturbation and the differences between the predictor function at its input and perturbed input.
  * **Monotonicity Metric Arya (Arya et al., 2019)**:
  * **[Monotonicity Metric Nguyen](https://arxiv.org/pdf/2007.07584.pdf) (Nguyen et al., 2020)**: The Nguyen Monotonicity measures the spearman rank correlation between the absolute values of the attribution and the uncertainty in the probability estimation.
  * **[Pixel Flipping](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (Bach et al., 2015)**: In Pixelflipping the impact of perturbing pixels in descending order according to the attributed value on the classification score is measured.
  * **[Region Perturbation](https://arxiv.org/pdf/1509.06321.pdf) (Samek et al., 2015)**: The region perturbation is the extension of Pixelflipping to flip an area rather than a single pixel.
  * **[Selectivity](https://arxiv.org/pdf/1706.07979.pdf) (Montavon et al., 2018)**: The selectivity measures how quickly an evaluated prediction function starts to drop when removing features with the highest attributed values.
  * **[SensitivityN](https://arxiv.org/pdf/1711.06104.pdf) (Ancona et al., 2019)**:
  * **[IROF](https://arxiv.org/pdf/2003.08747.pdf) (Iterative Removal of Features) (Rieger et al., 2020)**:
* *Robustness:*
  * **Continuity (Montavon et al., 2018)**:
  * **Input Independence Rate (Yang et al., 2019)**: input independence measures the percentage of inputs where a functionally insignificant patch (e.g., a dog) does not affect explanations significantly
  * **Local Lipschitz Estimate (Alvarez-Melis et al., 2018a, 2018b)**:
  * **Max-Sensitivity (Yeh et al., 2019)**:
  * **Avg-Sensitivity (Yeh et al., 2019)**:
* *Localisation:*
  * **[Pointing Game](https://link.springer.com/article/10.1007/s11263-017-1059-x) (Zhang et al., 2018)**: The Pointing Game checks, if the attribution with the highest score is located within the targeted object.
  * **[Attribution Localization](https://ieeexplore.ieee.org/abstract/document/9206975) (Kohlbrenner et al., 2020)**: The Attribution Localization measures the ratio of positive attributions within the targeted object towards the total positive attributions.
  * **[TKI](https://arxiv.org/pdf/2104.14995.pdf) (Theiner et al., 2021)**: The top-k intersection measures the intersection between a ground truth mask and the binarized explanation at the top k feature locations.
  * **[Relevance Rank Accuracy](https://arxiv.org/pdf/2003.07258.pdf) (Arras et al., 2021)**: The Relevance Rank Accuracy measures the ratio of highly attributed pixels within a ground-truth mask towards the size of the ground truth mask.
  * **[Relevance Mass Accuracy](https://arxiv.org/pdf/2003.07258.pdf) (Arras et al., 2021)**: The Relevance Mass Accuracy measures the ratio of positively attributed attributions inside the ground-truth mask towards the overall positive attributions.
  * **AUC ()**:
* *Complexity:*
  * **Sparseness (Chalasani et al., 2020)**:
  * **Complexity (Bhatt et al., 2020)**:
  * **Effective Complexity ()**:
* *Randomisation:*
  * **[Model Parameter Randomisation](https://proceedings.neurips.cc/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf) (Adebayo et al., 2018)**: The Model Parameter Randomization randomizes the parameters of single model layers in a cascading or independent way and measures the distance of the respective explanation to the original explanation
  * **[Random Logit](http://proceedings.mlr.press/v119/sixt20a.html) (Sixt et al., 2020)**: The Random Logit Measure is a meter for the distance between the original explanation and the explanation for a random other class.
* *Axiomatic:*
  * **Completeness (Sundararajan et al., 2017; **:
  * **Symmetry**: TBD
  * **(Non)Sensitivity**:
  * **Dummy**: TBD
  * **Input Invariance**: TBD

**Scope.** There is a couple of metrics that are popular but have not been included in the first version of the library. 
Metrics that require re-training of the network e.g., RoAR (Hooker et al., 2018) and Label Randomisation Test (Adebayo et al.,  2018) or rely on specifically designed datasets/ dataset modification e.g., Model Contrast Scores and Input Dependence Rate (Yang et al., 2019) and Attribution Percentage (Attr%) (Zhou et al., 2021) are considered out of scope of the first iteration.

It is worth nothing that this implementation primarily is motivated by image classification tasks. Further, it has been developed with attribution-based explanations in mind (which is a category of explanation methods that aim to assign an importance value to the model features and arguably, is the most studied kind of explanation). As a result, there will be both applications and explanation methods e.g., example-based methods where this library won't be useful.   

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

# You can use any function (not necessarily captum) to generate your explanations. 
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

* Compare explanation methods on different evalaution criteria (check out: `/tutorials/basic_example.ipynb`)
* Measure sensitivity of hyperparameter choice (check out: `/tutorials/hyperparameter_sensitivity.ipynb`)
* Understand how sensitivity of explanations change when a model is learning (check out: `/tutorials/training_robustness.ipynb`)
* Investigate to what extent metrics belonging to the same category score explanations similarliy (check out: `/tutorials/category_reliability.ipynb`)

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

## Cite this paper

To cite this paper use following Bibtex annotation:

	@misc{quantus,
	      title={Quantus: a metrics toolbox to evaluate neural network explanations}, 
	      author={},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}
