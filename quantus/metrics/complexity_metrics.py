from .base import Metric
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *


class Sparseness(Metric):
    """
    TODO. Rewrite docstring.
    Implementation of Sparseness metric by Chalasani et al., 2020.

    The sparseness test asks of explanations to have only the features that are
    truly predictive of the output F(x) should have significant contributions,
    and irrelevant or weakly-relevant features should have negligible
    contributions. It is quantified using the Gini Index applied to the
    vector of absolute values.

    References:
        1) Chalasani, Prasad, et al. "Concise explanations of neural
        networks using adversarial training." International Conference on Machine Learning. PMLR, 2020.
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.last_results = []
        self.all_results = []

        self.img_size = None
        self.nr_channels = None

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        **kwargs,
    ):

        if a_batch is None:
            a_batch = explain(
                model=model.to(kwargs.get("device", None)),
                inputs=x_batch,
                targets=y_batch,
                **kwargs,
            )

        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):
            # Based upon authors' implementation: https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py.
            a = np.abs(
                np.array(
                    np.reshape(a, (self.img_size * self.img_size,)),
                    dtype=np.float64,
                )
            )
            a += 0.0000001  # values canot be 0.
            a = np.sort(a)
            self.last_results.append(
                (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a))
                / (a.shape[0] * np.sum(a))
            )

        self.all_results.append(self.last_results)

        return self.last_results


class Complexity(Metric):
    """
    TODO. Rewrite docstring.
    Implementation of Complexity metric by Bhatt et al., 2020.

    "A complex explanation is one that uses all d features in its explanation of
    which features of x are important to f. Though this explanation may be
    faithful to the model (as defined above), it may be too difficult for the user
    to understand (especially if d is large.

    "Let Pg(i) denote the fractional contribution of feature xi to the total
    magnitude of the attribution. We define complexity as the entropy of Pg."

    References:
        1) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.last_results = []
        self.all_results = []

        self.img_size = None
        self.nr_channels = None

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        **kwargs,
    ):
        assert (
            "explanation_func" in kwargs
        ), "To evaluate with this metric, specify 'explanation_func' (str) e.g., 'Gradient'."

        if a_batch is None:
            a_batch = explain(
                model=model.to(kwargs.get("device", None)),
                inputs=x_batch,
                targets=y_batch,
                **kwargs,
            )

        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):
            a = (
                np.abs(
                    np.array(
                        np.reshape(a, (self.img_size * self.img_size,)),
                        dtype=np.float64,
                    )
                )
                / np.sum(np.abs(a))
            )

            self.last_results.append(scipy.stats.entropy(pk=a))

        self.all_results.append(self.last_results)

        return self.last_results


class EffectiveComplexity:
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.abs = self.kwargs.get("abs", True)
        self.eps = self.kwargs.get("eps", 1e-5)

        self.last_results = []
        self.all_results = []

        self.img_size = None
        self.nr_channels = None

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        **kwargs,
    ):
        assert (
            "explanation_func" in kwargs
        ), "To evaluate with this metric, specify 'explanation_func' (str) e.g., 'Gradient'."

        if a_batch is None:
            a_batch = explain(
                model=model.to(kwargs.get("device", None)),
                inputs=x_batch,
                targets=y_batch,
                **kwargs,
            )

        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = abs(a.flatten())

            self.last_results.append(int(np.sum(a > self.eps)))

        self.all_results.append(self.last_results)

        return self.last_results

if __name__ == '__main__':

    # Run tests!
    pass