## Quick installation

You can install Quantus in various ways. The different options are listed in the following.

### Installing via PyPI

If you already have [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org) installed on your machine, 
the most light-weight version of Quantus can be obtained from [PyPI](https://pypi.org/project/quantus/) as follows (no additional explainability functionality or deep learning framework will be included):

```setup
pip install quantus
```
Alternatively, you can simply add the desired deep learning framework (in brackets) to have the package installed together with Quantus.
To install Quantus with PyTorch, please run:
```setup
pip install "quantus[torch]"
```

For TensorFlow, please run:

```setup
pip install "quantus[tensorflow]"
```

### Installing via requirements.txt

Alternatively, you can simply install Quantus from the [requirements.txt](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/requirements.txt).
Note that this installation requires that either [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.TensorFlow.org) are already installed on your machine.

```setup
pip install -r requirements.txt
```

### Installing additional XAI Library support (PyPI only)

Most evaluation metrics in Quantus allow for a choice of either providing pre-computed explanations directly as an input, or instead making use of several wrappers implemented in `quantus.explain` around common explainability libraries. The
following XAI Libraries are currently supported:

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

Note that the three options above will also install the required frameworks (i.e., PyTorch or TensorFlow) respectively,
if they are not already installed in your environment. Note also, that not all explanation methods offered in **Captum** and **tf-explain**
 are included in `quantus.explain`.

### Installing tutorial requirements

The Quantus tutorials have more requirements than the base package, which you can install by running

```setup
pip install "quantus[tutorials]"
```

### Full installation

To simply install all of the above, you can run

```setup
pip install "quantus[full]"
```

### Package requirements

The package requirements are as follows:
```
python>=3.7.0
pytorch>=1.10.1
tensorflow==2.6.2
```