import numpy as np
from typing import Union
from .base import Metric
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *


class Continuity(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the Continuity test by Montavon et al., 2018.

    The test measures the strongest variation of the explanation in the input domain i.e.,
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    References:
        Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller.
        "Methods for interpreting and understanding deep neural networks."
        Digital Signal Processing 73 (2018): 1-15.

    Current assumptions:
        - that input dimensions is of equal size
        - made an quantitative interpretation of visually determining how similar f(x) and R(x1) curves are
        - using torch for explanation and model
    """

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        assert (
            kwargs.get("self.img_size", 224) % kwargs.get("nr_patches", 4) == 0
        ), "Set 'nr_patches' so that the modulo remainder returns 0 given the image size."

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)

        self.perturb_func = self.kwargs.get("perturb_func", translation_x_direction)
        self.similarity_func = self.kwargs.get("similarity_func", lipschitz_constant)

        self.nr_patches = self.kwargs.get("nr_patches", 4)
        self.patch_size = (self.img_size * 2) // self.nr_patches
        self.perturb_baseline = self.kwargs.get("perturb_baseline", 0.0)
        self.nr_steps = self.kwargs.get("nr_steps", 10)
        self.dx = self.img_size // self.nr_steps

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

        assert (
            "explanation_func" in kwargs
        ), "To run ContinuityTest specify 'explanation_func' (str) e.g., 'Gradient'."
        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            a_batch = explain(
                model.to(kwargs.get("device", None)),
                x_batch,
                y_batch,
                explanation_func=kwargs.get("explanation_func", "Gradient"),
                **kwargs,
            )

        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.last_results = {k: None for k in range(len(x_batch))}  # []

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = abs(a)

            sub_results = {k: [] for k in range(self.nr_patches + 1)}

            for step in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(
                    x,
                    **{
                        **{
                            "perturb_dx": (step + 1) * self.dx,
                            "perturb_baseline": self.perturb_baseline,
                        },
                        **self.kwargs,
                    },
                )
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    **kwargs,
                )

                # Store the prediction score as the last element of the sub_self.last_results dictionary.
                y_pred = float(
                    model(
                        torch.Tensor(x_perturbed)
                        .reshape(1, self.nr_channels, self.img_size, self.img_size)
                        .to(kwargs.get("device", None))
                    )[:, y]
                )
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

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """continuity_correlation_score"""
        return np.mean(
            [
                self.similarity_func(
                    self.last_results[sample][self.nr_patches],
                    self.last_results[sample][ix_patch],
                )
                for ix_patch in range(self.nr_patches)
                for sample in self.last_results.keys()
            ]
        )


class InputIndependenceRate(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the Input Independence Rate test by Yang et al., 2019.

    The test computes the input independence rate defined as the percentage of
    examples where the difference between x and x' is less than a threshold.

    References:
        Yang, Mengjiao, and Been Kim. "Benchmarking attribution methods with relative feature importance."
        arXiv preprint arXiv:1907.09701 (2019).

    Current assumptions:
        - that perturbed sample x' is "functionally insignificant" for the model

    TODO implementation:
        - optimization scheme for perturbing the image
        - double-check correctness of code interpretation (https://github.com/
        google-research-datasets/bam/blob/master/bam/metrics.py)

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)

        self.perturb_func = self.kwargs.get("perturb_func", None)
        self.similarity_func = self.kwargs.get("similarity_func", abs_difference)

        self.threshold = kwargs.get("threshold", 0.1)

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
        """


        Parameters
        ----------
        model
        x_batch
        y_batch
        a_batch
        kwargs

        Returns
        -------
        A list with a single float.

        """

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

        counts_thres = 0.0
        counts_corrs = 0.0

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = abs(a)

            # Generate explanation based on perturbed input x.
            x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
            a_perturbed = explain(
                model.to(kwargs.get("device", None)),
                x_perturbed,
                y,
                **kwargs,
            )
            y_pred = int(
                model(
                    torch.Tensor(x_perturbed)
                    .reshape(1, self.nr_channels, self.img_size, self.img_size)
                    .to(kwargs.get("device", None))
                )
                .max(1)
                .indices
            )

            if y_pred == y:
                counts_corrs += 1

                # Append similarity score.
                similarity = self.similarity_func(a.flatten(), a_perturbed.flatten())
                if similarity < self.threshold:
                    counts_thres += 1

        self.last_results.append(float(counts_thres / counts_corrs))
        self.all_results.append(self.last_results)

        return self.last_results


class LocalLipschitzEstimate(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of the Local Lipschitz Estimated (or Stability) test by Alvarez-Melis et al., 2018a, 2018b.

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
        - implement GP solver https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
        to more efficiently find max of sample distance

    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)

        self.perturb_func = self.kwargs.get("perturb_func", lipschitz_constant)
        self.similarity_func = self.kwargs.get("similarity_func", gaussian_noise)

        self.perturb_std = self.kwargs.get("perturb_std", 0.1)
        self.nr_steps = self.kwargs.get("nr_steps", 100)
        self.norm_numerator = self.kwargs.get("norm_numerator", distance_euclidean)
        self.norm_denominator = self.kwargs.get("norm_numerator", distance_euclidean)

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = abs(a)

            similarity_max = 0.0
            for i in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    **kwargs,
                )

                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed,
                )

                if similarity > similarity_max:
                    similarity_max = similarity

            # Append similarity score.
            self.last_results.append(similarity_max)

        self.all_results.append(self.last_results)

        return self.last_results


class MaxSensitivity(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of max-sensitivity of an explanation by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuing how explanations
    change under slight perturbation.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

    Note that Similar to EstimatedLocalLipschitzConstant, but may be considered more robust.
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)

        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)

        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)

        # self.agg_func = self.kwargs.get("agg_func", np.max)

        self.std = self.kwargs.get("perturb_radius", 0.2)
        self.nr_steps = self.kwargs.get("nr_steps", 200)

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = abs(a)

            sensitivities_norm_max = 0.0
            for _ in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)

                # TODO. Kwargs need to have a callable called explanation_func ...
                # Update on all Robustness metrics.
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    **kwargs,
                )

                sensitivities = self.similarity_func(
                    a=a.flatten(), b=a_perturbed.flatten()
                )
                sensitivities_norm = self.norm_numerator(
                    a=sensitivities
                ) / self.norm_denominator(a=x.flatten())

                if sensitivities_norm > sensitivities_norm_max:
                    sensitivities_norm_max = sensitivities_norm

            # Append max sensitivity score.
            self.last_results.append(sensitivities_norm_max)

        self.all_results.append(self.last_results)

        return self.last_results


class AvgSensitivity(Metric):
    """
    TODO. Rewrite docstring.

    Implementation of avg-sensitivity of an explanation by Yeh at el., 2019.

    Using Monte Carlo sampling-based approximation while measuing how explanations
    change under slight perturbation.

    References:
        1) Yeh, Chih-Kuan, et al. "On the (in) fidelity and sensitivity for explanations."
        arXiv preprint arXiv:1901.09392 (2019).
        2) Bhatt, Umang, Adrian Weller, and José MF Moura. "Evaluating and aggregating
        feature-based model explanations." arXiv preprint arXiv:2005.00631 (2020).

    Note that Similar to EstimatedLocalLipschitzConstant, but may be considered more robust.
    """

    def __init__(self, *args, **kwargs):

        super(Metric, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", False)

        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)

        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)

        # self.agg_func = self.kwargs.get("agg_func", np.max)

        self.std = self.kwargs.get("perturb_radius", 0.2)
        self.nr_steps = self.kwargs.get("nr_steps", 200)

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

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = abs(a.flatten())

            self.temp_results = []
            for _ in range(self.nr_steps):
                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain(
                    model.to(kwargs.get("device", None)),
                    x_perturbed,
                    y,
                    **kwargs,
                )

                sensitivities = self.similarity_func(
                    a=a.flatten(), b=a_perturbed.flatten()
                )
                sensitivities_norm = self.norm_numerator(
                    a=sensitivities
                ) / self.norm_denominator(a=x.flatten())

                self.temp_results.append(sensitivities_norm)

            # Append average sensitivity score.
            self.last_results.append(float(np.mean(self.temp_results)))

        self.all_results.append(self.last_results)

        return self.last_results


if __name__ == '__main__':

    # Run tests!
    pass