from sklearn.metrics import auc
import numpy as np
from .base import Measure
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similarity_func import *
from ..helpers.explanation_func import *


class FaithfulnessTest(Measure):
    """
    Implements basis functionality for the following evaluation measures:

        - Infidelity (Yeh et al., 2019)
        - Monotonicity (Luss et al., 2019), (Nguyen et al., 2020)
        - Smallest Sufficient Region (SSR), Smallest Destroying Region (SDR) (Dabkowski et al., 2017)
        - Faithfulness correlation (Bhatt et al., 2020)
        - Pixel-flipping and its variations:
            - Pixel-flipping (Bach et al., 2015)
            - Region perturbation (Samek et a., 2016)
            - IROF test (Rieger et al., 2018)
            - Selectivity (Montavon et al., 2018)
        - Sensitivity-n (Ancona et al., 2018)
        - Faithfulness Estimate (Alvarez-Melis et al., 2018)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.pixels_in_step = self.kwargs.get("pixels_in_step", 128)
        assert (
                           self.img_size * self.img_size) % self.pixels_in_step == 0, "Set 'pixels_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                               self.img_size * self.img_size) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.pixels_in_step = (self.img_size * self.img_size) / self.max_steps_per_input

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            a = abs(a.flatten())

            # Get indices of sorted attributions (descending).
            a_indices = np.argsort(-a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None)))[:, y])

            pred_deltas = []
            att_sums = []

            # Create n masked versions of input x.
            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[(self.pixels_in_step * i_ix):(self.pixels_in_step * (i_ix + 1))]
                x_perturbed = self.perturb_func(img=x.flatten(),
                                                **{"index": a_ix, "perturb_baseline": self.perturb_baseline})

                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(model(
                        torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None)))[:, y])
                pred_deltas.append(float(y_pred - y_pred_i))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        return results


class FaithfulnessEstimate(Measure):
    """
    Implementation of Faithfulness Estimate by Alvares-Melis at el., 2018.

    Computes the correlations of probability drops and the relevance scores on various points,
    and show the aggregate statistics.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).

    Current assumptions:
        - We are iterating over 128 pixels at a time, to reduce the number of steps x nr_pixels
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.pixels_in_step = self.kwargs.get("pixels_in_step", 1)
        assert (
                           self.img_size * self.img_size) % self.pixels_in_step == 0, "Set 'pixels_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                               self.img_size * self.img_size) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.pixels_in_step = (self.img_size * self.img_size) / self.max_steps_per_input

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            a = abs(a.flatten())
            a_indices = np.argsort(-a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None)))[:, y])

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[(self.pixels_in_step * i_ix):(self.pixels_in_step * (i_ix + 1))]
                x_perturbed = self.perturb_func(img=x.flatten(),
                                                **{"index": a_ix, "perturb_baseline": self.perturb_baseline})
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(model(
                        torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None)))[:, y])
                pred_deltas.append(float(y_pred - y_pred_i))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        return results


class Infidelity(Measure):
    """
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

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", mse)
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)
        self.perturb_patch_sizes = self.kwargs.get("perturb_patch_sizes", list(np.arange(10, 30)))

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        # Remove patch sizes that are not compatible with input size.
        self.perturb_patch_sizes = [i for i in self.perturb_patch_sizes if self.img_size % i == 0]

        self.pixels_in_step = self.kwargs.get("pixels_in_step", 128)
        assert (
                       self.img_size * self.img_size) % self.pixels_in_step == 0, "Set 'pixels_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                           self.img_size * self.img_size) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.pixels_in_step = (self.img_size * self.img_size) / self.max_steps_per_input

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(torch.nn.Softmax()(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None))))[:, y])

            sub_results = []
            for patch_size in self.perturb_patch_sizes:

                att_sums = np.zeros((int(a.shape[0] / patch_size), int(a.shape[1] / patch_size)))
                pred_deltas = np.zeros((int(a.shape[0] / patch_size), int(a.shape[1] / patch_size)))

                a = abs(a)

                for i_x, top_left_x in enumerate(range(0, x.shape[1], patch_size)):

                    for i_y, top_left_y in enumerate(range(0, x.shape[2], patch_size)):

                        # Sum attributions for patch.
                        att_sums[i_x][i_y] = a[top_left_x: top_left_x + patch_size,
                                            top_left_y: top_left_y + patch_size
                                            ].sum()

                        # Perturb input.
                        x_temp = x.copy()
                        x_perturbed = self.perturb_func(x_temp,
                                                        **{"patch_size": patch_size,
                                                           "nr_channels": self.nr_channels,
                                                           "perturb_baseline": self.perturb_baseline,
                                                           "top_left_y": top_left_y,
                                                           "top_left_x": top_left_x,
                                                           })

                        # Predict on perturbed input x.
                        with torch.no_grad():
                            y_pred_i = float(torch.nn.Softmax()(model(torch.Tensor(x_perturbed)
                                                   .reshape(1, self.nr_channels, self.img_size, self.img_size)
                                                   .to(kwargs.get("device", None))))[:, y])

                        pred_deltas[i_x][i_y] = float(y_pred - y_pred_i)

                sub_results.append(self.similarity_func(a=att_sums.flatten(), b=pred_deltas.flatten()))

            results.append(np.mean(sub_results))

        return results


class MonotonicityMetric(Measure):
    """
    Implementation of Montonicity Metric by Nguyen at el., 2020.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Nguyen, An-phi, and María Rodríguez Martínez. "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Luss, Ronny, et al. "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).

    Todo. Double-check Luss interpretation; does it align with Nguyen implementation?
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_spearman)
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.pixels_in_step = self.kwargs.get("pixels_in_step", 1)
        assert (
                       self.img_size * self.img_size) % self.pixels_in_step == 0, "Set 'pixels_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                           self.img_size * self.img_size) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.pixels_in_step = (self.img_size * self.img_size) / self.max_steps_per_input

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            a = abs(a.flatten())
            a_indices = np.argsort(a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None)))[:, y])

            atts = []
            preds = []

            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[(self.pixels_in_step * i_ix):(self.pixels_in_step * (i_ix + 1))]
                x_perturbed = self.perturb_func(img=x.flatten(),
                                                **{"index": a_ix, "perturb_baseline": self.perturb_baseline})
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(model(
                        torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None)))[:, y])

                atts.append(float(sum(a[a_ix])))
                preds.append(y_pred_i)

            results.append(self.similarity_func(a=atts, b=preds)) # np.all(np.diff(y_pred_i[a_indices]) >= 0)

        return results


class PixelFlipping(Measure):
    """
    Implementation of Pixel-Flipping experiment by Bach at el., 2015.

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
        self.args = args
        self.kwargs = kwargs
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_indices)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.pixels_in_step = self.kwargs.get("pixels_in_step", 1)
        assert (
                       self.img_size * self.img_size) % self.pixels_in_step == 0, "Set 'pixels_in_step' so that the modulo remainder returns 0 given the image size."
        self.max_steps_per_input = self.kwargs.get("max_steps_per_input", None)

        if self.max_steps_per_input is not None:
            assert (
                           self.img_size * self.img_size) % self.max_steps_per_input == 0, "Set 'max_steps_per_input' so that the modulo remainder returns 0 given the image size."
            self.pixels_in_step = (self.img_size * self.img_size) / self.max_steps_per_input

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            a = abs(a.flatten())
            a_indices = np.argsort(a)

            preds = []
            x_perturbed = x.copy().flatten()

            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                # Perturb input by indices of attributions.
                a_ix = a_indices[(self.pixels_in_step * i_ix):(self.pixels_in_step * (i_ix + 1))]
                x_perturbed = self.perturb_func(img=x_perturbed,
                                                **{"index": a_ix, "perturb_baseline": self.perturb_baseline})
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(torch.nn.Softmax()(model(torch.Tensor(x_perturbed)
                                                              .reshape(1, self.nr_channels, self.img_size,
                                                                       self.img_size)
                                                              .to(kwargs.get("device", None))))[:, y])
                preds.append(y_pred_i)

            results.append(preds)

        return results


class RegionPerturbation(Measure):
    """
    Implementation of Region Perturbation by Samek at el., 2015.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    Done according to Most Relevant First (MoRF) and Area Over the Perturbation Curve
    (AOPC).

    References:
        1) Samek, Wojciech, et al. "Evaluating the visualization of what a deep
         neural network has learned." IEEE transactions on neural networks and
          learning systems 28.11 (2016): 2660-2673.

    Current assumptions:
        - 9 x 9 patch sizes was used in the paper as regions, but using 8 x 8
        to make sure non-overlapping
        - they called it "area over the MoRF perturbation curve" but for me it
        looks like a simple deduction of function outputs?
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_patch)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "uniform")

        self.regions_evaluation = self.kwargs.get("regions_evaluation", 100)
        self.patch_size = self.kwargs.get("patch_size", 8)
        self.random_order = self.kwargs.get("random_order", False)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        # Assert that patch size that are not compatible with input size.
        assert self.img_size % self.patch_size == 0, "Set 'patch_size' so that the modulo remainder returns 0 given the image size."

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = {k: None for k in range(len(x_batch))}

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.nn.Softmax()(torch.Tensor(x)
                                       .reshape(1, self.nr_channels, self.img_size, self.img_size)
                                       .to(kwargs.get("device", None))))[:, y])

            # Get patch indices of sorted attributions (descending).
            a = abs(a)
            att_sums = np.zeros((int(a.shape[0] / self.patch_size), int(a.shape[1] / self.patch_size)))
            x_perturbed = x.copy()
            patches = []
            sub_results = []

            # DEBUG.
            # plt.imshow(a.reshape(224, 224), cmap="seismic")
            # plt.show()

            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):
                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[top_left_x: top_left_x + self.patch_size,
                                         top_left_y: top_left_y + self.patch_size
                                         ].sum()
                    patches.append([top_left_y, top_left_x])

            patch_order = {k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)}

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(min(self.regions_evaluation, len(patch_order))):

                # Calculate predictions on a random order.
                if self.random_order == True:
                    order = random.choice(patch_order)
                    top_left_y = patch_order[k][0]
                    top_left_x = patch_order[k][1]
                else:
                    top_left_y = patch_order[k][0]
                    top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(x_perturbed,
                                                **{"patch_size": self.patch_size,
                                                   "nr_channels": self.nr_channels,
                                                   "perturb_baseline": self.perturb_baseline,
                                                   "top_left_y": top_left_y,
                                                   "top_left_x": top_left_x,
                                                   })

                # DEBUG.
                # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                # plt.show()

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_i = float(torch.nn.Softmax()(model(torch.Tensor(x_perturbed)
                                                              .reshape(1, self.nr_channels, self.img_size,
                                                                       self.img_size)
                                                              .to(kwargs.get("device", None))))[:, y])
                sub_results.append(y_pred - y_pred_i)

            results[sample] = sub_results

        print(f"AOPC (MoRF): {self.calculate_area_over_perturbation_curve(results):.4f}")

        return results


    def calculate_area_over_perturbation_curve(self, results: list):
        """Calculate the area over the perturbation curve AOPC."""
        return np.mean([np.array(results[sample]) / self.regions_evaluation for sample in
                        results.keys()])  # area = trapz(y, dx=1000)



class Selectivity(Measure):
    """
    Implementation of Selectivity test by Montavan at el., 2018.

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
        self.args = args
        self.kwargs = kwargs
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_patch)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 32)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        # Assert that patch size that are not compatible with input size.
        assert self.img_size % self.patch_size == 0, "Set 'patch_size' so that the modulo remainder returns 0 given the image size."

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = {k: None for k in range(len(x_batch))}

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.nn.Softmax()(torch.Tensor(x)
                                       .reshape(1, self.nr_channels, self.img_size, self.img_size)
                                       .to(kwargs.get("device", None))))[:, y])

            # Get patch indices of sorted attributions (descending).
            a = abs(a)
            att_sums = np.zeros((int(a.shape[0] / self.patch_size), int(a.shape[1] / self.patch_size)))
            x_perturbed = x.copy()
            patches = []
            sub_results = []

            # DEBUG.
            # plt.imshow(a.reshape(224, 224), cmap="seismic")
            # plt.show()

            for i_x, top_left_x in enumerate(range(0, x.shape[1], self.patch_size)):
                for i_y, top_left_y in enumerate(range(0, x.shape[2], self.patch_size)):
                    # Sum attributions for patch.
                    att_sums[i_x][i_y] = a[top_left_x: top_left_x + self.patch_size,
                                         top_left_y: top_left_y + self.patch_size
                                         ].sum()
                    patches.append([top_left_y, top_left_x])

            patch_order = {k: v for k, v in zip(np.argsort(att_sums, axis=None)[::-1], patches)}

            # Increasingly perturb the input and store the decrease in function value.
            for k in range(len(patch_order)):
                top_left_y = patch_order[k][0]
                top_left_x = patch_order[k][1]

                x_perturbed = self.perturb_func(x_perturbed,
                                                **{"patch_size": self.patch_size,
                                                   "nr_channels": self.nr_channels,
                                                   "perturb_baseline": self.perturb_baseline,
                                                   "top_left_y": top_left_y,
                                                   "top_left_x": top_left_x,
                                                   })

                # DEBUG.
                # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                # plt.show()

                # Predict on perturbed input x and store the difference from predicting on unperturbed input.
                with torch.no_grad():
                    y_pred_i = float(torch.nn.Softmax()(model(torch.Tensor(x_perturbed)
                                                              .reshape(1, self.nr_channels, self.img_size,
                                                                       self.img_size)
                                                              .to(kwargs.get("device", None))))[:, y])
                sub_results.append(y_pred_i)

            results[ix] = sub_results

        print(f"AUC: {self.calculate_area_under_the_curve(results):.4f}")

        return results


    def calculate_area_under_the_curve(self, results: list):
        """Calculate the area under the curve."""
        return np.mean([np.trapz(results[sample], dx=1000) for sample in results.keys()])


class SensitivityN(Measure):
    """
    Implementation of Sensitivity-N test by Ancona at el., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions
    for any subset of features of cardinality n is equal to the variation of the output
    Sc caused removing the features in the subset. The test computes the
    correlation between sum of attributions and delta output.

    ...Pearson correlation coefficient (PCC) computed between the sum of the attributions
    and the variation in the target output varying n from one to about 80% of the total
    number of features. The PCC is averaged across a thousand of samples from each dataset.
    The sampling is performed using a uniform probability distribution over the features,
    given that we assume no prior knowledge on the correlation between them. This allows
    to apply this evaluation not only to images but to any kind of input.

    References:
        1) Ancona, Marco, et al. "Towards better understanding of gradient-based attribution
        methods for deep neural networks." arXiv preprint arXiv:1711.06104 (2017).

    Current assumptions:
         - In the paper, they showcase a MNIST experiment where
         4x4 patches with black baseline value. Since we are taking ImageNet as dataset,
         we take 224/28=8 i.e., 8 times bigger patches to replicate the same analysis
         - Also, instead of replacing with a black pixel we take the mean of the
         neighborhood, so not to distort the image distribution completely.

    """

    def __init__(self, *args, **kwargs):

        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get("perturb_func", baseline_replacement_by_patch)
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.patch_size = self.kwargs.get("patch_size", 32)
        self.n_max_percentage = self.kwargs.get("n_max_percentage", 0.8)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.max_features = int(self.n_max_percentage * self.img_size * self.img_size)

        super(Measure, self).__init__()


    def __call__(
            self,
            model,
            x_batch: np.array,
            y_batch: Union[np.array, int],
            a_batch: Union[np.array, None],
            **kwargs
    ):
        assert (
                "explanation_func" in kwargs
        ), "To run RobustnessTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
                np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )

        results = []

        for x, y, a in zip(x_batch, y_batch, a_batch):

            # Get indices of sorted attributions (descending).
            a = abs(a.flatten())
            a_indices = np.argsort(-a)

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None)))[:, y])

            pred_deltas = []
            att_sums = []

            for i_ix, a_ix in enumerate(np.arange(1, self.max_features)): # a_indices[::self.pixels_in_step]

                # Perturb input by indices of attributions.
                a_ix = a_indices[(self.pixels_in_step * i_ix):(self.pixels_in_step * (i_ix + 1))]
                x_perturbed = self.perturb_func(img=x.flatten(),
                                                **{"index": a_ix, "perturb_baseline": self.perturb_baseline})
                # Predict on perturbed input x.
                with torch.no_grad():
                    y_pred_i = float(model(
                        torch.Tensor(x_perturbed)
                            .reshape(1, self.nr_channels, self.img_size, self.img_size)
                            .to(kwargs.get("device", None)))[:, y])
                pred_deltas.append(float(y_pred - y_pred_i))

                # Sum attributions.
                att_sums.append(a[a_ix].sum())

            results.append(self.similarity_func(a=att_sums, b=pred_deltas))

        return results


    def calculate_area_under_the_curve(self, results: list):
        """Calculate the area under the curve."""
        return np.mean([np.trapz(results[sample], dx=1000) for sample in results.keys()])
