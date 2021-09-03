"""This module contains the collection of axiomatic metrics to evaluate attribution-based explanations of neural network models."""
from .base import Metric
from ..helpers.utils import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *


class Completeness(Metric):
    """
    TODO. Rewrite docstring.
    Implementation of Completeness test by Sundararajan et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018.

    Attribution completeness asks that the total attribution is proportional to
    the explainable evidence at the output/ or some function of the model output

    References:
        1)
        2)
        3)
        4)

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.output_transformation_func = self.kwargs.get(
            "output_transformation_func", lambda x: x
        )

        self.last_results = []
        self.all_results = []

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

            if self.abs:
                a = np.abs(a)

            if np.sum(a) == self.output_transformation_func(y):
                self.last_results.append(True)
            else:
                self.last_results.append(False)

        self.all_results.append(self.last_results)

        return self.last_results


class Symmetry(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    pass


class InputInvariance(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    pass


class Sensitivity(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    pass


class Dummy(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    pass

if __name__ == '__main__':

    # Run tests!
    pass