"""This module contains the collection of complexity metrics to evaluate attribution-based explanations of neural network models."""
from typing import Union, List, Dict
from .base import Metric
from ..helpers.utils import *
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *
from ..helpers.normalise_func import *
from ..helpers.warn_func import *


class Sparseness(Metric):
    """
    Implementation of Sparseness metric by Chalasani et al., 2020.

    The sparseness test asks of explanations to have only the features that are
    truly predictive of the output F(x) should have significant contributions,
    and irrelevant or weakly-relevant features should have negligible
    contributions. It is quantified using the Gini Index applied to the
    vector of absolute values.

    References:
        1) Chalasani, Prasad, et al. "Concise explanations of neural networks using adversarial training."
        International Conference on Machine Learning. PMLR, 2020.

    Assumptions:
        Based on authors' implementation as found on the following link:
        https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py.

    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Sparseness metric is likely to be sensitive to the choice of normalising 'normalise' (and "
            "'normalise_func') and if taking absolute values of attributions 'abs'. "
            "\nGo over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. \nTo view all relevant hyperparameters call .get_params of the "
            "metric instance. \nFor further reading, please see: Chalasani, Prasad, et al. Concise explanations of "
            "neural networks using adversarial training.' International Conference on Machine Learning. PMLR, 2020."
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        warn_parameterisation(text=self.text_warning)
        warn_attributions(normalise=self.normalise, abs=self.abs)


    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)
            else:
                a = np.abs(a)
                print(
                    "An absolute operation is applied on the attributions (regardless of the 'abs' parameter value)"
                    "since it is required by the metric."
                )

            if self.normalise:
                a = self.normalise_func(a)

            a = np.array(
                np.reshape(a, (self.img_size * self.img_size,)), dtype=np.float64
            )
            a += 0.0000001
            a = np.sort(a)
            self.last_results.append(
                (np.sum((2 * np.arange(1, a.shape[0] + 1) - a.shape[0] - 1) * a))
                / (a.shape[0] * np.sum(a))
            )

        self.all_results.append(self.last_results)

        return self.last_results


class Complexity(Metric):
    """
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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Complexity metric is likely to be sensitive to the choice of normalising 'normalise' (and "
            "'normalise_func') and if taking absolute values of attributions 'abs'. "
            "\nGo over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. \nTo view all relevant hyperparameters call .get_params of the "
            "metric instance. \nFor further reading, please see: Bhatt, Umang, Adrian Weller, and José MF Moura. "
            "'Evaluating and aggregating feature-based model explanations.' arXiv preprint arXiv:2005.00631 (2020)\n"
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        warn_parameterisation(text=self.text_warning)
        warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)
            else:
                a = np.abs(a)
                print(
                    "An absolute operation is applied on the attributions (regardless of the 'abs' parameter value)"
                    "since it is required by the metric."
                )

            if self.normalise:
                a = self.normalise_func(a)

            a = (
                np.array(
                    np.reshape(a, (self.img_size * self.img_size,)),
                    dtype=np.float64,
                )
                / np.sum(np.abs(a))
            )

            self.last_results.append(scipy.stats.entropy(pk=a))

        self.all_results.append(self.last_results)

        return self.last_results


class EffectiveComplexity(Metric):
    """ """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.eps = self.kwargs.get("eps", 1e-5)
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.text_warning = (
            "\nThe Effective complexity metric is likely to be sensitive to the choice of threshold 'eps'. "
            "\nGo over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. \nTo view all relevant hyperparameters call .get_params of the "
            "metric instance. \nFor further reading, please see: Nguyen, An-phi, and María Rodríguez Martínez. 'On "
            "quantitative aspects of model interpretability.' arXiv preprint arXiv:2007.07584 (2020)."
        )
        self.last_results = []
        self.all_results = []

        # Asserts and warnings.
        warn_parameterisation(text=self.text_warning)
        warn_attributions(normalise=self.normalise, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[int]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = self.kwargs.get("explain_func", Callable)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a.flatten())
            else:
                a = np.abs(a.flatten())
                print(
                    "An absolute operation is applied on the attributions (regardless of the 'abs' parameter value)"
                    "since it is required by the metric."
                )

            if self.normalise:
                a = self.normalise_func(a)

            self.last_results.append(int(np.sum(a > self.eps)))  # int operation?

        self.all_results.append(self.last_results)

        return self.last_results


if __name__ == "__main__":

    # Run tests!
    pass
