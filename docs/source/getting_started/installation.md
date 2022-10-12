## Quick Installation

### Installing from PyPI

If you already have `Pytorch` or `Tensorflow` installed on your machine, Quantus can be obtained from [PyPI](https://pypi.org/project/quantus/) as follows:

```setup
pip install quantus
```

Otherwise, you can simply add the desired framework in brackets, and it will be installed in addition to Quantus:

```setup
pip install quantus[torch]
```

OR

```setup
pip install quantus[tensorflow]
```

### Installing from requirements.txt

Alternatively, you can simply install from the requirements.txt found [here](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/requirements.txt),
however, this only installs with the default setup, requiring either `PyTorch` or `Tensorflow`:

```setup
pip install -r requirements.txt
```

### Installing XAI Library Support (PyPI only)

Most evaluation metrics in Quantus allow for a choice of either providing pre-computed explanations directly as an input,
or to instead make use of several wrappers implemented in `quantus.explain` around common explainability libraries.
The following XAI Libraries are currently supported:

**Captum**

To enable the use of wrappers around [Captum](https://captum.ai/), you need to have `PyTorch` already installed and can then run

```setup
pip install quantus[extras]
```

**tf-explain**

To enable the use of wrappers around [tf.explain](https://github.com/sicara/tf-explain), you need to have `Tensorflow` already installed and can then run

```setup
pip install quantus[extras]
```

**Zennit**

To use Quantus with support for the [Zennit](https://github.com/chr5tphr/zennit) library you need to have `PyTorch` already installed and can then run

```setup
pip install quantus[zennit]
```

### Package Requirements

```
python>=3.7.0
pytorch>=1.10.1
tensorflow==2.6.2
tqdm==4.62.3
```