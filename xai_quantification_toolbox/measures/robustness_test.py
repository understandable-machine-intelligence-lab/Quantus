import numpy as np
from typing import Union
from .base import Measure
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similarity_func import *
from ..helpers.explanation_func import *


class RobustnessTest(Measure):
    """
    Implements basis functionality for the following evaluation measures:

        • Continuity (Montavon et al., 2018)
        • IIR (Yang et al., 2019)
        • Estimated Lipschitz constant (Alvarez-Melis, 2019)
        • avg-Sensitivity, max-Sensitivity (Yeh et al., 2019)
        • Stability (Alvarez-Melis, 2018)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.perturb_func = self.kwargs.get("perturb_func", gaussian_noise)
        self.similarity_func = self.kwargs.get("similarity_func", distance_euclidean)

        super(RobustnessTest, self).__init__()

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
            # model.attribute(batch=x, neuron_selection=y, explanation_func=kwargs.get("explanation_func", "gradient"),)

        results = []
        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Generate explanation based on perturbed input x.
            x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
            a_perturbed = explain(
                model.to(kwargs.get("device", None)),
                x_perturbed,
                y,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
            # model.attribute(batch=x_perturbed, neuron_selection=y, explanation_func=kwargs.get("explanation_func", "gradient"),)

            # Append similarity score.
            results.append(
                self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.cpu().numpy().flatten(),
                    c=x.flatten(),
                    d=x_perturbed,
                )
            )

        return results


class ContinuityTest(Measure):
    """
    Implementation of the Continuity test by Montavon et al (2018).

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    References:
        Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Current assumptions:
        • that input is squared
        • made an quantitative interpretation of visually determining how similar f(x) and R(x1) curves are
        • using torch for explanation and model
    """

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        assert (
            kwargs.get("img_size", 224) % kwargs.get("nr_patches", 4) == 0
        ), "Set 'nr_patches' so that the modulo remainder returns 0 given the image size."

        self.args = args
        self.kwargs = kwargs

        self.perturb_func = self.kwargs.get(
            "perturb_func", translation_x_direction
        )
        self.similarity_func = self.kwargs.get(
            "similarity_func", lipschitz_constant
        )

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.nr_patches = self.kwargs.get("nr_patches", 4)
        self.patch_size = (self.img_size * 2) // self.nr_patches
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)
        self.nr_steps = self.kwargs.get("nr_steps", 10)
        self.dx = self.img_size // self.nr_steps

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
        ), "To run ContinuityTest specify 'explanation_func' (str) e.g., 'Gradient'."
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
            # model.attribute(batch=x, neuron_selection=y, explanation_func=kwargs.get("explanation_func", "gradient"),)

        results = []  # {k: None for k in range(len(x_batch))}

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            sub_results = {k: [] for k in range(self.nr_patches + 1)}

            for step in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(
                    x,
                    **{
                        "perturb_dx": (step + 1) * self.dx,
                        "perturb_baseline": self.perturb_baseline,
                    }
                )
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    explanation_func=kwargs.get("explanation_func", "Gradient"),
                    device=kwargs.get("device", None),
                )
                # model.attribute(batch=x_perturbed, neuron_selection=y, explanation_func=kwargs.get("explanation_func", "gradient"),)

                # DEBUG.
                # a_perturbed_test = np.ones_like(a_perturbed.cpu().numpy())
                # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                # plt.show()

                # Store the prediction score as the last element of the sub_results dictionary.
                y_pred = model(
                    torch.Tensor(x_perturbed)
                    .reshape(1, self.nr_channels, self.img_size, self.img_size)
                    .to(kwargs.get("device", None))
                )[:, y]
                sub_results[self.nr_patches].append(y_pred)

                ix_patch = 0
                for i_x, top_left_x in enumerate(
                    range(0, self.img_size, self.patch_size)
                ):
                    for i_y, top_left_y in enumerate(
                        range(0, self.img_size, self.patch_size)
                    ):
                        # Sum attributions for patch.
                        patch_sum = float(
                            a_perturbed[
                                :,
                                top_left_x : top_left_x + self.patch_size,
                                top_left_y : top_left_y + self.patch_size,
                            ]
                            .abs()
                            .sum()
                        )
                        sub_results[ix_patch].append(patch_sum)
                        ix_patch += 1

                        # DEBUG.
                        # a_perturbed_test[:, top_left_x: top_left_x + self.patch_size, top_left_y: top_left_y + self.patch_size] = 0
                        # plt.imshow(a_perturbed_test.reshape(224, 224))
                        # plt.show()

                ix_patch = 0

            # Append similarity score.
            results.append(
                np.mean(
                    [
                        self.similarity_func(
                            sub_results[self.nr_patches], sub_results[ix_patch]
                        )
                        for ix_patch in range(self.nr_patches)
                    ]
                )
            )  # results[ix] = sub_results

        return results


class InputIndependenceRate(RobustnessTest):
    """
    Implementation of the Input Independence Rate test by Yang et al (2019).

    The test computes the input independence rate defined as the percentage of
    examples where the difference between x and x' is less than a threshold.

    References:
        Yang, Mengjiao, and Been Kim. "Benchmarking attribution methods with relative feature importance."
        arXiv preprint arXiv:1907.09701 (2019).

    Current assumptions:
        • that perturbed sample x' is "functionally insignificant" for the model

    TODO implementation:
        • optimization scheme for perturbing the image
        • double-check correctness of code interpretation (https://github.com/
        google-research-datasets/bam/blob/master/bam/metrics.py)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.perturb_func = self.kwargs.get("perturb_func", None)
        self.similarity_func = self.kwargs.get("similarity_func", abs_difference)

        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_channels = self.kwargs.get("nr_channels", 3)

        self.threshold = kwargs.get("threshold", 0.1)

        super(RobustnessTest, self).__init__()

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

        counts_thres = 0.0
        counts_corrs = 0.0
        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Generate explanation based on perturbed input x.
            x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
            a_perturbed = explain(
                model.to(kwargs.get("device", None)),
                x_perturbed,
                y,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                device=kwargs.get("device", None),
            )
            y_pred = int(model(
                torch.Tensor(x_perturbed)
                    .reshape(1, self.nr_channels, self.img_size, self.img_size)
                    .to(kwargs.get("device", None))
            ).max(1).indices)

            if (y_pred == y):
                counts_corrs += 1

                # Append similarity score.
                similarity = self.similarity_func(a.flatten(), a_perturbed.cpu().numpy().flatten())
                if similarity < self.threshold:
                    counts_thres += 1

        return float(counts_thres / counts_corrs)



class LocalLipschitzEstimate(RobustnessTest):
    """
    Implementation of the Local Lipschitz Estimated (or Stability) test by Alvarez-Melis et al (2018a, 2018b).

    This tests asks how consistent are the explanations for similar/neighboring examples.
    The test denotes a (weaker) empirical notion of stability based on discrete,
    finite-sample neighborhoods i.e., argmax_(||f(x) - f(x')||_2 / ||x - x'||_2)
    where f(x) is the explanation for input x and x' is the perturbed input.

    References:
        1) Alvarez-Melis, David, and Tommi S. Jaakkola. "On the robustness of interpretability methods."
        arXiv preprint arXiv:1806.08049 (2018).

        2) Alvarez-Melis, David, and Tommi S. Jaakkola. "Towards robust interpretability with self-explaining
        neural networks." arXiv preprint arXiv:1806.07538 (2018).


    TODO implementation:
        • implement GP solver https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
        to more efficiently find max of sample distance

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.perturb_func = self.kwargs.get("perturb_func", lipschitz_constant)
        self.similarity_func = self.kwargs.get("similarity_func", gaussian_noise)

        self.std = self.kwargs.get("std", 0.1)
        self.nr_samples = self.kwargs.get("nr_samples", 100)
        self.norm_numerator = self.kwargs.get("norm_numerator", distance_euclidean)
        self.norm_denominator = self.kwargs.get("norm_numerator", distance_euclidean)

        super(RobustnessTest, self).__init__()

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

            similarity_max = 0.0
            for i in range(self.nr_samples):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    explanation_func=kwargs.get("explanation_func", "Gradient"),
                    device=kwargs.get("device", None),
                )

                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.cpu().numpy().flatten(),
                    c=x.flatten(),
                    d=x_perturbed,
                )

                if similarity > similarity_max:
                    similarity_max = similarity

            # Append similarity score.
            results.append(similarity_max)

        return results


class SensitivityMax(RobustnessTest):
    """
    Implementation of max-sensitivity of an explanation by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuing how explanations
    change under slight perturbation.

    References:
        Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).

    Note that Similar to EstimatedLocalLipschitzConstant, but may be considered more robust.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)

        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)

        #self.agg_func = self.kwargs.get("agg_func", np.max)

        self.std = self.kwargs.get("perturb_radius", 0.2)
        self.nr_samples = self.kwargs.get("nr_samples", 10)

        super(RobustnessTest, self).__init__()

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

            sensitivities_norm_max = 0.0
            for i in range(self.nr_samples):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    explanation_func=kwargs.get("explanation_func", "Gradient"),
                    device=kwargs.get("device", None),
                )

                sensitivities = self.similarity_func(a=a.flatten(),
                                                     b=a_perturbed.cpu().numpy().flatten())
                sensitivities_norm = self.norm_numerator(a=sensitivities) / self.norm_denominator(a=x.flatten())

                if sensitivities_norm > sensitivities_norm_max:
                    sensitivities_norm_max = sensitivities_norm

            # Append similarity score.
            results.append(sensitivities_norm_max)

        return results
