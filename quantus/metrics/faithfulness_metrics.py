from sklearn.metrics import auc
import numpy as np
from .base import Metric
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *


class FaithfulnessCorrelation(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of faithfulness correlation by Bhatt et al., 2020.

    For each test sample, |S| features are randomly selected and replace them
    with baseline values (zero baseline or average of set). Thereafter,
    Pearson’s correlation coefficient between the predicted logits of each
    modified test point and the average explanation attribution for only the
    subset of features is calculated. Results is average over multiple runs and
    several test samples.

    References:
        1) Bhatt ...

    Current assumptions:

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")

        self.subset_size = self.kwargs.get("subset_size", 50)
        self.nr_runs = self.kwargs.get("nr_runs", 100)

        self.last_results = []
        self.all_results = []

        self.text = f"""\n\nThe Faithfulness Correlation metric is known to be sensitive to the choice of baseline value 'perturb_baseline', size of subset |S| 'subset_size' and the number of runs (for each input and explanation pair) 'nr_runs'. \nGo over and select each hyperparameter of the SelectivityN metric carefully to avoid misinterpretation of scores. \nTo view all relevant hyperparameters call list_hyperparameters method. \nFor more reading, please see [INSERT CITATION]."""

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        **kwargs,
    ):

        # Generate warning.
        self.print_warning(text=self.text)

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            if self.abs:
                a = abs(a.flatten())

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None))
                    )[:, y]
                )

            logit_deltas = []
            att_means = []

            # For each test data point, execute a couple of runs.
            for i_ix in range(self.nr_runs):

                # Randomly mask by subset size.
                a_ix = np.random.choice(a.shape[0], self.subset_size, replace=False)
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                )

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        model(
                            torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )[:, y]
                    )

                logit_deltas.append(float(y_pred - y_pred_perturb))

                # Average attributions of the subset.
                att_means.append(np.mean(a[a_ix]))

            self.last_results.append(self.similarity_func(a=att_means, b=logit_deltas))

        self.last_results = [np.mean(self.last_results)]
        self.all_results.append(self.last_results)

        return self.last_results


class FaithfulnessEstimate(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018a and 2018b.

    Computes the correlations of probability drops and the relevance scores on various points,
    showing the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).

    Current additions:
        - To reduce the number of steps x nr_pixels we iterate/ remove 128 pixels at a time

    Current assumptions:
        - Choice of correlation metric; Pearson correlation coefficient.

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")

        self.features_in_step = self.kwargs.get("features_in_step", 1)
        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(-a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None))
                    )[:, y]
                )

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                )
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        model(
                            torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )[:, y]
                    )
                pred_deltas.append(float(y_pred - y_pred_perturb))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            self.last_results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        self.all_results.append(self.last_results)

        return self.last_results

    def plot_aggregated_statistics(self):
        """Plot aggregated statistics of faithfulness scores."""


class Infidelity(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of infidelity by Yeh at el., 2019.

    "Explanation infidelity represents the expected mean-squared error between the explanation
    multiplied by a meaningful input perturbation and the differences between the predictor function
    at its input and perturbed input."

    "Our infidelity measure is defined as the expected difference between the two terms:
    (a) the dot product of the input perturbation to the explanation and
    (b) the output perturbation (i.e., the difference in function values after significant
    perturbations on the input)."

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).

    Current assumptions:
        - original implementation support perturbation of Gaussian noise and using squares
            "We thus propose a modified subset distribution from that described in Proposition 2.5
            where the perturbation Z has a uniform distribution over square patches with predefined
            length, which is in spirit similar to the work of [49]"
            (https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py)
        - assumes inputs are squared

    # TODO. Verify implementation - I^T x attr.
    # TODO. Implement multipy_by_inputs.
    # TODO. Normalization implement.

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.loss_func = self.kwargs.get("loss_func", mse)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.perturb_patch_sizes = self.kwargs.get(
            "perturb_patch_sizes", list(np.arange(10, 30))
        )
        self.n_perturb_samples = self.kwargs.get("n_perturb_samples", 10)
        self.multipy_by_inputs = self.kwargs.get("multipy_by_inputs", False)
        self.normalise = self.kwargs.get("normalise", False)
        # self.baseline = self.kwargs.get("baseline", "black")

        self.img_size = kwargs.get("img_size", 224)
        self.nr_channels = kwargs.get("nr_channels", 3)

        # Remove patch sizes that are not compatible with input size.
        self.perturb_patch_sizes = [
            i for i in self.perturb_patch_sizes if self.img_size % i == 0
        ]

        self.features_in_step = self.kwargs.get("features_in_step", 128)
        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )
                    )[:, y]
                )

            sub_sub_results = []

            for _ in range(self.n_perturb_samples):

                sub_results = []

                for patch_size in self.perturb_patch_sizes:

                    pred_deltas = np.zeros(
                        (int(a.shape[0] / patch_size), int(a.shape[1] / patch_size))
                    )

                    a = abs(a)

                    for i_x, top_left_x in enumerate(range(0, x.shape[1], patch_size)):

                        for i_y, top_left_y in enumerate(
                            range(0, x.shape[2], patch_size)
                        ):

                            # Perturb input.
                            x_temp = x.copy()
                            x_perturbed = self.perturb_func(
                                x_temp,
                                **{
                                    "patch_size": patch_size,
                                    "nr_channels": self.nr_channels,
                                    "perturb_baseline": self.perturb_baseline,
                                    "top_left_y": top_left_y,
                                    "top_left_x": top_left_x,
                                },
                            )

                            if self.multipy_by_inputs:
                                pass  # x - x_perturbed
                            else:
                                pass  # (x - x_perturbed) / (x - self.baselines)

                            # Predict on perturbed input x.
                            with torch.no_grad():
                                y_pred_perturb = float(
                                    torch.nn.Softmax()(
                                        model(
                                            torch.Tensor(x_perturbed)
                                            .reshape(
                                                1,
                                                self.nr_channels,
                                                self.img_size,
                                                self.img_size,
                                            )
                                            .to(kwargs.get("device", None))
                                        )
                                    )[:, y]
                                )

                            pred_deltas[i_x][i_y] = float(y_pred - y_pred_perturb)

                    sub_results.append(None)
                    # self.loss_func(torch.mul(a=a.flatten(), np.transpose(x_perturbed).flatten()), b=pred_deltas.flatten())

                sub_sub_results.append(np.mean(sub_results))

            self.last_results.append(np.mean(sub_sub_results))

        self.all_results.append(self.last_results)

        return self.last_results


class Monotonicity1(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Montonicity Metric by Arya at el., 2019.

    Montonicity tests if adding more positive evidence increases the probability of classification in the specified
    class.

    References:
        1) Arya, Vijay, et al. "One explanation does not fit all: A toolkit and taxonomy of ai explainability
        techniques." arXiv preprint arXiv:1909.03012 (2019).
        2) Luss, Ronny, et al. "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).

    TODO. Double-check Luss interpretation; does it align with Nguyen implementation?
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")

        self.features_in_step = self.kwargs.get(
            "features_in_step",
        )
        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(a)
            preds = []

            x_perturbed = (
                torch.Tensor()
                .new_full(size=x.shape, fill_value=self.perturb_baseline)
                .numpy()
                .flatten()
            )

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x_perturbed,
                    **{"index": a_ix, "replacement_values": x.flatten()[a_ix]},
                )

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )

                preds.append(y_pred_perturb)

            self.last_results.append(np.all(np.diff(preds[a_indices]) >= 0))

        self.all_results.append(self.last_results)

        return self.last_results


class Monotonicity2(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Montonicity Metric by Nguyen at el., 2020.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).

    TODO. Change the change in probability drop with the uncertainty in prediction.

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        1
        self.features_in_step = self.kwargs.get(
            "features_in_step",
        )

        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(a)

            atts = []
            preds = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                )

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )

                atts.append(float(sum(a[a_ix])))
                preds.append(y_pred_perturb)

            self.last_results.append(self.similarity_func(a=atts, b=preds))

        self.all_results.append(self.last_results)

        return self.last_results


class PixelFlipping(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Pixel-Flipping experiment by Bach et al., 2015.

    The basic idea is to compute a decomposition of a digit for a digit class
    and then flip pixels with highly positive, highly negative scores or pixels
    with scores close to zero and then to evaluate the impact of these flips
    onto the prediction scores (mean prediction is calculated).

    References:
        1) Bach, Sebastian, et al. "On pixel-wise explanations for non-linear classifier
        decisions by layer-wise relevance propagation." PloS one 10.7 (2015): e0130140.

    Current assumptions:
        - Using 8 pixel at a time instead of one single pixel when we use ImageNet.
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_indices
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")

        self.features_in_step = self.kwargs.get("features_in_step", 1)
        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = []

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

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(a)

            preds = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x_perturbed,
                    **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                )
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )
                preds.append(y_pred_perturb)

            self.last_results.append(preds)

        self.all_results.append(self.last_results)

        return self.last_results


class RegionPerturbation(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Region Perturbation by Samek et al., 2015.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
         neural network has learned." IEEE transactions on neural networks and
          learning systems 28.11 (2016): 2660-2673.

    Current assumptions:
        -Done according to Most Relevant First (MoRF) and Area Over the Perturbation Curve
        (AOPC).
        - 9 x 9 patch sizes was used in the paper as regions, but using 8 x 8
        to make sure non-overlapping
        - they called it "area over the MoRF perturbation curve" but for me it
        looks like a simple deduction of function outputs?

    # TODO. Make curves relative to baseline (computed as the AOPC curves for random heatmaps (i.e., random ordering O)).
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")

        self.regions_evaluation = self.kwargs.get("regions_evaluation", 100)
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.random_order = self.kwargs.get("random_order", False)
        self.order = self.kwargs.get("order", "MoRF").lower()

        # Assert that patch size that are not compatible with input size.
        assert (
            self.img_size % self.patch_size == 0
        ), "Set 'patch_size' so that the modulo remainder returns 0 given the image size."

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
        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = {k: None for k in range(len(x_batch))}

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                **kwargs,
            )

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.nn.Softmax()(
                            torch.Tensor(x)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )
                    )[:, y]
                )

            # Get patch indices of sorted attributions (descending).
            a = abs(a)
            att_sums = np.zeros(
                (int(a.shape[0] / self.patch_size), int(a.shape[1] / self.patch_size))
            )
            x_perturbed = x.copy()
            patches = []
            sub_results = []

            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):
                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[
                        top_left_x : top_left_x + self.patch_size,
                        top_left_y : top_left_y + self.patch_size,
                    ].sum()
                    patches.append([top_left_y, top_left_x])

            if self.order == "morf":
                # Order attributions according to the most relevant first.
                patch_order = {
                    k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)
                }
            else:
                patch_order = {
                    k: v for k, v in zip(np.argsort(att_sums, axis=None), patches)
                }
                # Order attributions according to the least relevant first.

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(min(self.regions_evaluation, len(patch_order))):

                # Calculate predictions on a random order.
                if self.random_order == True:
                    order = random.randint(0, len(patch_order))
                    top_left_y = patch_order[order][0]
                    top_left_x = patch_order[order][1]
                else:
                    top_left_y = patch_order[k][0]
                    top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(
                    x_perturbed,
                    **{
                        "patch_size": self.patch_size,
                        "nr_channels": self.nr_channels,
                        "perturb_baseline": self.perturb_baseline,
                        "top_left_y": top_left_y,
                        "top_left_x": top_left_x,
                    },
                )

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )
                sub_results.append(y_pred - y_pred_perturb)

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """Calculate the area over the perturbation curve AOPC."""
        return np.mean(
            [
                np.array(self.last_results[sample]) / self.regions_evaluation
                for sample in self.last_results.keys()
            ]
        )  # area = trapz(y, dx=1000)


class Selectivity(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Selectivity test by Montavan et al., 2018.

    At each iteration, a patch of size 4 x 4 corresponding to the region with
    highest relevance is set to black. The plot keeps track of the function value
    as the features are being progressively removed and computes an average over
     a large number of examples.

    References:
        1) Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         * Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 32)

        # Assert that patch size that are not compatible with input size.
        assert (
            self.img_size % self.patch_size == 0
        ), "Set 'patch_size' so that the modulo remainder returns 0 given the image size."

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
        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = {k: None for k in range(len(x_batch))}

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                **kwargs,
            )

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(
                    model(
                        torch.nn.Softmax()(
                            torch.Tensor(x)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )
                    )[:, y]
                )

            # Get patch indices of sorted attributions (descending).
            a = abs(a)
            att_sums = np.zeros(
                (int(a.shape[0] / self.patch_size), int(a.shape[1] / self.patch_size))
            )
            x_perturbed = x.copy()
            patches = []
            sub_results = []

            # DEBUG.
            # plt.imshow(a.reshape(224, 224), cmap="seismic")
            # plt.show()

            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):
                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[
                        top_left_x : top_left_x + self.patch_size,
                        top_left_y : top_left_y + self.patch_size,
                    ].sum()
                    patches.append([top_left_y, top_left_x])

            patch_order = {
                k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)
            }

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(len(patch_order)):
                top_left_y = patch_order[k][0]
                top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(
                    x_perturbed,
                    **{
                        "patch_size": self.patch_size,
                        "nr_channels": self.nr_channels,
                        "perturb_baseline": self.perturb_baseline,
                        "top_left_y": top_left_y,
                        "top_left_x": top_left_x,
                    },
                )

                # DEBUG.
                # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                # plt.show()

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_perturb = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )
                sub_results.append(y_pred_perturb)

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """Calculate the area under the curve."""
        return np.mean(
            [
                np.trapz(self.last_results[sample], dx=1000)
                for sample in self.last_results.keys()
            ]
        )


class SensitivityN(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is repoted. Sampling is performed using a uniform probability distribution over the features.

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         - Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.
         - I don't get why they have so high correlation in the paper, maybe using a better baseline_value?
         - Also I don't get why correlation is only reported positive?

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get(
            "perturb_func", baseline_replacement_by_patch
        )
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")
        self.n_max_percentage = self.kwargs.get("n_max_percentage", 0.8)

        self.features_in_step = self.kwargs.get("features_in_step", 28)
        self.max_features = int(
            (0.8 * self.img_size * self.img_size) // self.features_in_step
        )

        assert (
            self.img_size * self.img_size
        ) % self.features_in_step == 0, "Set 'features_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                self.img_size * self.img_size
            ) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.features_in_step = (
                self.img_size * self.img_size
            ) / self.max_steps_per_input

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
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_result = []

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                **kwargs,
            )

        sub_results_pred_deltas = {k: [] for k in range(len(x_batch))}
        sub_results_att_sums = {k: [] for k in range(len(x_batch))}

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(-a)

            # Predict on x.
            with torch.no_grad():
                y_pred = float(
                    torch.nn.Softmax()(
                        model(
                            torch.Tensor(x)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None))
                        )
                    )[:, y]
                )

            att_sums = []
            pred_deltas = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                if i_ix <= self.max_features:

                    # Perturb input by indices of attributions.
                    a_ix = a_indices[
                        (self.features_in_step * i_ix) : (
                            self.features_in_step * (i_ix + 1)
                        )
                    ]
                    x_perturbed = self.perturb_func(
                        img=x_perturbed,
                        **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                    )

                    # DEBUG.
                    # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                    # plt.show()

                    # Sum attributions.
                    att_sums.append(float(a[a_ix].sum()))

                    with torch.no_grad():
                        y_pred_perturb = float(
                            torch.nn.Softmax()(
                                model(
                                    torch.Tensor(x_perturbed)
                                    .reshape(
                                        1,
                                        self.nr_channels,
                                        self.img_size,
                                        self.img_size,
                                    )
                                    .to(kwargs.get("device", None))
                                )
                            )[:, y]
                        )
                    pred_deltas.append(y_pred - y_pred_perturb)

            sub_results_att_sums[sample] = att_sums
            sub_results_pred_deltas[sample] = pred_deltas

        # Rearange sublists.
        sub_results_pred_deltas_l = {k: [] for k in range(self.max_features)}
        sub_results_att_sums_l = {k: [] for k in range(self.max_features)}

        for k in range(self.max_features):
            for sublist1 in list(sub_results_pred_deltas.values()):
                sub_results_pred_deltas_l[k].append(sublist1[k])
            for sublist2 in list(sub_results_att_sums.values()):
                sub_results_att_sums_l[k].append(sublist2[k])

        # Measure similarity for each n.
        self.last_result = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(self.max_features)
        ]
        self.all_results.append(self.last_result)

        return self.last_result


class IROF(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of IROF (Iterative Removal of Features) by Rieger at el., 2020.

    Description.

    References:
        1)


    Todo. Write test.

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Get indices of sorted attributions (descending).
            if self.abs:
                a = abs(a.flatten())
            a_indices = np.argsort(a)

            preds = []

            for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[
                    (self.features_in_step * i_ix) : (
                        self.features_in_step * (i_ix + 1)
                    )
                ]
                x_perturbed = self.perturb_func(
                    img=x.flatten(),
                    **{"index": a_ix, "perturb_baseline": self.perturb_baseline},
                )
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(
                        torch.nn.Softmax()(
                            model(
                                torch.Tensor(x_perturbed)
                                .reshape(
                                    1, self.nr_channels, self.img_size, self.img_size
                                )
                                .to(kwargs.get("device", None))
                            )
                        )[:, y]
                    )
                preds.append(float(y_pred_i))

            self.last_results.append(preds)

        return self.last_results


if __name__ == '__main__':

    # Run tests!
    pass
