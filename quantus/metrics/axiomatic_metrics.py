"""This module contains the collection of axiomatic metrics to evaluate attribution-based explanations of neural network models."""
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


class Completeness(Metric):
    """
    Implementation of Completeness test by Sundararajan et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018.

    Attribution completeness asks that the total attribution is proportional to
    the explainable evidence at the output/ or some function of the model output.

    References:
        1) Completeness - Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks."
         International Conference on Machine Learning. PMLR, 2017.
        2) Summation to delta - Shrikumar, Avanti, Peyton Greenside, and Anshul Kundaje. "Learning important
        features through propagating activation differences." International Conference on Machine Learning. PMLR, 2017.
        3) Conservation - Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting
        and understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Assumptions:
        This implementation does completeness test against logits, not softmax.
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.output_func = self.kwargs.get("output_func", lambda x: x)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.text_warning = (
            "\nThe Completeness metric is likely to be sensitive to the choice of "
            "baseline value 'perturb_baseline' and the function to modify the model response 'output_func'. "
            "\nGo over and select each hyperparameter of the metric carefully to "
            "avoid misinterpretation of scores. \nTo view all relevant hyperparameters call .get_params of the "
            "metric instance. \nFor further reading, please see: Completeness - Sundararajan, Mukund, Ankur Taly, "
            "and Qiqi Yan. 'Axiomatic attribution for deep networks.' International Conference on "
            "Machine Learning. PMLR, 2017."
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
    ) -> List[bool]:

        # Update kwargs.
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
        assert_attributions(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            x_perturbed = x.flatten()
            x_perturbed = self.perturb_func(
                img=x_perturbed,
                **{
                    "index": np.arange(x, len(x)),
                    "perturb_baseline": self.perturb_baseline,
                },
            )

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            # Predict on baseline.
            with torch.no_grad():
                y_pred_baseline = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(self.kwargs.get("device", None))
                    )[:, y]
                )

            if np.sum(a) == self.output_func(y_pred - y_pred_baseline):
                self.last_results.append(True)

            else:
                self.last_results.append(False)

        self.all_results.append(self.last_results)

        return self.last_results


class NonSensitivity(Metric):
    """
    Implementation of NonSensitivity by Nguyen at el., 2020.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
    """

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.eps = self.kwargs.get("eps", 1e-5)
        self.n_samples = self.kwargs.get("n_samples", 100)
        self.abs = self.kwargs.get("abs", True)
        self.normalise = self.kwargs.get("normalise", True)
        self.normalise_func = self.kwargs.get("normalise_func", normalise_by_max)
        self.default_plot_func = Callable
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.text_warning = (
            "\nThe Non-sensitivity metric is likely to be sensitive to the choice of "
            "baseline value 'perturb_baseline', the number of samples to iterate over 'n_samples' and the threshold"
            " value function for the feature to be considered having an insignificant contribution to the model. "
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
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
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
        assert_attributions(a_batch=a_batch, x_batch=x_batch)

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = np.abs(a)

            if self.normalise:
                a = self.normalise_func(a)

            non_features = set(list(np.argwhere(a).flatten() < self.eps))

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
                        y_pred_perturbed = float(
                            torch.nn.Softmax()(
                                model(
                                    torch.Tensor(x_perturbed)
                                    .reshape(
                                        1,
                                        self.nr_channels,
                                        self.img_size,
                                        self.img_size,
                                    )
                                    .to(self.kwargs.get("device", None))
                                )
                            )[:, y]
                        )
                        preds.append(y_pred_perturbed)

                    vars.append(np.var(preds))

            non_features_vars = set(list(np.argwhere(vars).flatten() < self.eps))
            self.last_results.append(
                len(non_features_vars.symmetric_difference(non_features))
            )

        self.all_results.append(self.last_results)

        return self.last_results


if __name__ == "__main__":

    # Run tests!
    pass
