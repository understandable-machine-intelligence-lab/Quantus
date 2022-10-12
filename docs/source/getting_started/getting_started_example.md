## Getting started

To use the library, you'll need a couple of ingredients; a model, some input data and labels (to be explained).
In this example, we use `torch` but we also support evaluation of `tensorflow` models.

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


### Misc functionality

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
