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

        • Infidelity (Yeh et al., 2019)
        • Monotonicity (Luss et al., 2019), (Nguyen et al., 2020)
        • Smallest Sufficient Region (SSR), Smallest Destroying Region (SDR) (Dabkowski et al., 2017)
        • Faithfulness correlation (Bhatt et al., 2020)
        • Pixel-flipping and its variations:
            • Original (Bach et al., 2015)
            • Region segmentation (Samek et a., 2016)
            • IROF test (Rieger et al., 2018)
            • Selectivity (Montavon et al., 2018)
        • Sensitivity-n (Ancona et al., 2018)
        • Faithfulness Estimate (Alvarez-Melis et al., 2018)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.similarity_func = self.kwargs.get("similarity_func", correlation_pearson)
        self.perturb_func = self.kwargs.get("perturb_func", replacement_by_indices)
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

        super(FaithfulnessTest, self).__init__()

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

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
            att_sum = []

            ### GET PERTURBATIONS: Create n masked versions of input x.
            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                if i_ix == 0:
                    a_ix = a_indices[:self.pixels_in_step]
                else:
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
                att_sum.append(a[a_ix].sum())

            results.append(self.similarity_func(a=att_sum, b=pred_deltas))

        return results


class FaithfulnessEstimate(FaithfulnessTest):
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
        • We are iterating over 128 pixels at a time, to reduce the number of steps x nr_pixels
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

        super(FaithfulnessTest, self).__init__()

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

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
            att_sum = []

            for i_ix, a_ix in enumerate(a_indices[::self.pixels_in_step]):

                if i_ix == 0:
                    a_ix = a_indices[:self.pixels_in_step]
                else:
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
                att_sum.append(a[a_ix].sum())

            results.append(self.similarity_func(a=att_sum, b=pred_deltas))

        return results


class Infidelity(FaithfulnessTest):
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
        Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).

    Current assumptions:
        • original implementation support perturbation of Gaussian noise and using squares
            "We thus propose a modified subset distribution from that described in Proposition 2.5
            where the perturbation Z has a uniform distribution over square patches with predefined
            length, which is in spirit similar to the work of [49]"
            (https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py)
        • assumes inputs are squared

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

        super(FaithfulnessTest, self).__init__()

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Predict on input.
            with torch.no_grad():
                y_pred = float(model(
                    torch.Tensor(x)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None)))[:, y])

            sub_results = []
            for patch_size in self.perturb_patch_sizes:

                att_sum = np.zeros((int(a.shape[0] / patch_size), int(a.shape[1] / patch_size)))
                pred_deltas = np.zeros((int(a.shape[0] / patch_size), int(a.shape[1] / patch_size)))

                a = abs(a)

                for i_x, top_left_x in enumerate(range(0, x.shape[1], patch_size)):

                    for i_y, top_left_y in enumerate(range(0, x.shape[2], patch_size)):
                        # Sum attributions for patch.
                        att_sum[i_x][i_y] = a[top_left_x: top_left_x + patch_size,
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
                            y_pred_i = float(model(torch.Tensor(x_perturbed)
                                                   .reshape(1, self.nr_channels, self.img_size, self.img_size)
                                                   .to(kwargs.get("device", None)))[:, y])

                        pred_deltas[i_x][i_y] = float(y_pred - y_pred_i)

                sub_results.append(self.similarity_func(a=att_sum.flatten(), b=pred_deltas.flatten()))

            results.append(np.mean(sub_results))

        return results