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
    # TODO. Adapt with baseline.

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.output_func = self.kwargs.get("output_func", lambda x: x)
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

        # Update kwargs.
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = kwargs.get("explain_func", None)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_atts(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                        model(
                            torch.Tensor(x)
                                .reshape(1, self.nr_channels, self.img_size, self.img_size)
                                .to(self.kwargs.get("device", None))
                        )[:, y]
                    #torch.nn.Softmax()()
                )

            #res[m] = np.float(np.abs(target_value - baseline - np.sum(attributions)))
            #if np.abs(target_value) > 0:
                #res[m + '_relative'] = float(res[m] / np.abs(target_value))

            if np.sum(a) == self.output_func(y_pred):
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


class NonSensitivity(Metric):
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
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
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

        # Update kwargs.
        self.kwargs = {**kwargs, **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]}}
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        if a_batch is None:

            # Asserts.
            explain_func = kwargs.get("explain_func", None)
            assert_explain_func(explain_func=explain_func)

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_atts(a_batch=a_batch, x_batch=x_batch)


        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            non_features = set(list(np.argwhere(a).flatten() < self.eps).flatten())

            vars = []
            for a_i in range(len(a)):

                preds = []
                for _ in range(self.n_samples):

                    x_perturbed = self.perturb_func(
                        img=x.flatten(),
                        **{"index": a_i, "perturb_baseline": self.perturb_baseline},
                    )

                    # Predict on perturbed input x.
                    with torch.no_grad():
                        y_pred_perturbed = float(torch.nn.Softmax()(
                        model(torch.Tensor(x_perturbed)
                              .reshape(1, self.nr_channels, self.img_size, self.img_size).to(self.kwargs.get("device", None))))[:, y])
                        preds.append(y_pred_perturbed)

                    vars.append(np.var(preds))

            non_features_vars = set(list(np.argwhere(vars).flatten() < self.eps).flatten())
            self.last_results.append(len(non_features_vars.symmetric_difference(non_features)))

        self.all_results.append(self.last_results)

        return self.last_results


class Dummy(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Implement metric.
    """

    pass

if __name__ == '__main__':

    # Run tests!
    pass