# Getting started

The following will give a short introduction for how to get started with Quantus.

**Note**: This example is based on the [PyTorch](https://pytorch.org/) framework, but we also support 
[Tensorflow](https://www.tensorflow.org), which would differ only in the {ref}`preliminaries <prelim>` 
(i.e., the model and data loading), 
as well as in the available XAI libraries.

## Preliminaries
(prelim)=
Quantus implements methods for the quantitative evaluation of XAI methods.
Generally, in order to apply these, you will need:
* A model
* Input data (and labels)
* Explanations to evaluate

### Model and Data

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

### Explanations

We still need some explanations to evaluate, however. 
For this, there are two possibilities in Quantus:

#### Using Pre-computed Explanations
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

#### Using Quantus XAI Wrappers
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

## Evaluating Explanations with Quantus

### Quantus Metrics

Quantus implements XAI evaluation metrics from different categories 
(faithfulness, localisation, robustness, ...) which all inherit from the base `quantus.Metric` class. 

Metrics are designed as `Callables`. To apply a metric to your setting (e.g., [Max-Sensitivity](https://arxiv.org/abs/1901.09392)), 
they first need to be instantiated

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

### Customizing Metrics

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
* Hyperparameters affecting the inputs (data, model, explanations) to each metric are set in the `__call__` method of each metric
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


### Large-Scale Evaluations
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

## Extending Quantus

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