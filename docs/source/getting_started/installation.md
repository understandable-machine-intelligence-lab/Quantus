## Quick Installation

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

**Package requirements**

```
python>=3.7.0
pytorch>=1.10.1
tensorflow==2.6.2
tqdm==4.62.3
```