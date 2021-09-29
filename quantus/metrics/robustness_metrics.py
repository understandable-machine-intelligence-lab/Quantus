"""This module contains the collection of robustness metrics to evaluate attribution-based explanations of neural network models."""
import numpy as np
from typing import Union, List, Dict
from .base import Metric
from ..helpers.utils import *
from ..helpers.asserts import *
from ..helpers.plotting import *
from ..helpers.norm_func import *
from ..helpers.perturb_func import *
from ..helpers.similar_func import *
from ..helpers.explanation_func import *
from ..helpers.normalize_func import *


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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.perturb_std = self.kwargs.get("perturb_std", 0.1)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", distance_euclidean)
        self.norm_denominator = self.kwargs.get("norm_numerator", distance_euclidean)
        self.perturb_func = self.kwargs.get("perturb_func", lipschitz_constant)
        self.similarity_func = self.kwargs.get("similarity_func", gaussian_noise)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalize:
            warn_normalize_abs(normalize=self.normalize, abs=self.abs)

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
        self.last_result = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Get explanation function and make asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            similarity_max = 0.0
            for i in range(self.nr_samples):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain_func(
                    model=model, inputs=x_perturbed, targets=y, **self.kwargs
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)
                if self.normalize:
                    a_perturbed = self.normalize_func(a_perturbed)

                # Measure similarity.
                similarity = self.similarity_func(
                    a=a.flatten(),
                    b=a_perturbed.flatten(),
                    c=x.flatten(),
                    d=x_perturbed.flatten(),
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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", False)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.std = self.kwargs.get("perturb_radius", 0.2)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalize:
            warn_normalize_abs(normalize=self.normalize, abs=self.abs)

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
        self.last_result = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Get explanation function and make asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            sensitivities_norm_max = 0.0
            for _ in range(self.nr_samples):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain_func(
                    model=model, inputs=x_perturbed, targets=y, **self.kwargs
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalize:
                    a_perturbed = self.normalize_func(a_perturbed)

                # Measure sensitivity.
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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", False)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.std = self.kwargs.get("perturb_radius", 0.2)
        self.nr_samples = self.kwargs.get("nr_samples", 200)
        self.norm_numerator = self.kwargs.get("norm_numerator", fro_norm)
        self.norm_denominator = self.kwargs.get("norm_denominator", fro_norm)
        self.perturb_func = self.kwargs.get("perturb_func", uniform_sampling)
        self.similarity_func = self.kwargs.get("similarity_func", difference)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalize:
            warn_normalize_abs(normalize=self.normalize, abs=self.abs)

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
        self.last_result = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            self.temp_results = []
            for _ in range(self.nr_samples):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
                a_perturbed = explain_func(
                    model=model, inputs=x_perturbed, targets=y, **self.kwargs
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalize:
                    a_perturbed = self.normalize_func(a_perturbed)

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


class Continuity(Metric):
    """
    TODO. Rewrite docstring.
    TODO. Fix this to work with != 4 patches. Can't get why?

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

    @attributes_check
    def __init__(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.img_size = self.kwargs.get("img_size", 224)
        self.nr_patches = self.kwargs.get("nr_patches", 4)
        self.patch_size = (self.img_size * 2) // self.nr_patches
        self.perturb_baseline = self.kwargs.get("perturb_baseline", "black")
        self.nr_steps = self.kwargs.get("nr_steps", 28)
        self.dx = self.img_size // self.nr_steps
        self.perturb_func = self.kwargs.get("perturb_func", translation_x_direction)
        self.similarity_func = self.kwargs.get("similarity_func", lipschitz_constant)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalize:
            warn_normalize_abs(normalize=self.normalize, abs=self.abs)
        assert_patch_size(patch_size=self.patch_size, img_size=self.img_size)


    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> Dict[int, List[float]]:

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_results = {k: None for k in range(len(x_batch))}

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        for sample, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

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

                # Generate explanations on perturbed input.
                a_perturbed = explain_func(
                    model=model, inputs=x_perturbed, targets=y, **self.kwargs
                )

                if self.abs:
                    a_perturbed = np.abs(a_perturbed)

                if self.normalize:
                    a_perturbed = self.normalize_func(a_perturbed)

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
                        a_perturbed_patch = a_perturbed[
                            :,
                            top_left_x : top_left_x + self.patch_size,
                            top_left_y : top_left_y + self.patch_size,
                        ]
                        if self.abs:
                            a_perturbed_patch = np.abs(a_perturbed_patch.flatten())

                        if self.normalize:
                            a_perturbed_patch = self.normalize_func(
                                a_perturbed_patch.flatten()
                            )

                        # DEBUG.
                        # a_perturbed[:,
                        # top_left_x: top_left_x + self.patch_size,
                        # top_left_y: top_left_y + self.patch_size,] = 0
                        # plt.imshow(a_perturbed.reshape(224, 224))
                        # plt.show()

                        # Sum attributions for patch.
                        patch_sum = float(sum(a_perturbed_patch))
                        sub_results[ix_patch].append(patch_sum)
                        ix_patch += 1

            self.last_results[sample] = sub_results

        self.all_results.append(self.last_results)

        return self.last_results

    @property
    def aggregated_score(self):
        """Implements a continuity correlation score (an addition to original method) to evaluate the
        relationship between change in explanation and change in function output."""
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

    @attributes_check
    def __init__(self, *args, **kwargs):

        super().__init__()

        self.args = args
        self.kwargs = kwargs
        self.abs = self.kwargs.get("abs", True)
        self.normalize = self.kwargs.get("normalize", True)
        self.normalize_func = self.kwargs.get("normalize_func", normalize_by_max)
        self.default_plot_func = Callable
        self.threshold = kwargs.get("threshold", 0.1)
        self.perturb_func = self.kwargs.get("perturb_func", Callable)
        self.similarity_func = self.kwargs.get("similarity_func", abs_difference)
        self.last_results = []
        self.all_results = []

        # Asserts and checks.
        if self.abs or self.normalize:
            warn_normalize_abs(normalize=self.normalize, abs=self.abs)

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        *args,
        **kwargs,
    ) -> List[float]:
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

        # Update kwargs.
        self.nr_channels = kwargs.get("nr_channels", np.shape(x_batch)[1])
        self.img_size = kwargs.get("img_size", np.shape(x_batch)[-1])
        self.kwargs = {
            **kwargs,
            **{k: v for k, v in self.__dict__.items() if k not in ["args", "kwargs"]},
        }
        self.last_result = []

        # Get explanation function and make asserts.
        explain_func = self.kwargs.get("explain_func", Callable)
        assert_explain_func(explain_func=explain_func)

        if a_batch is None:

            # Generate explanations.
            a_batch = explain_func(
                model=model,
                inputs=x_batch,
                targets=y_batch,
                **self.kwargs,
            )

        # Get explanation function and make asserts.
        assert_attributions(x_batch=x_batch, a_batch=a_batch)

        counts_thres = 0.0
        counts_corrs = 0.0

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            if self.abs:
                a = np.abs(a)

            if self.normalize:
                a = self.normalize_func(a)

            # Generate explanation based on perturbed input x.
            x_perturbed = self.perturb_func(x.flatten(), **self.kwargs)
            a_perturbed = explain_func(
                model=model, x_batch=x_perturbed, y_batch=y_batch, **self.kwargs
            )

            if self.abs:
                a_perturbed = np.abs(a_perturbed)

            if self.normalize:
                a_perturbed = self.normalize_func(a_perturbed)

            y_pred_perturbed = int(
                model(
                    torch.Tensor(x_perturbed)
                    .reshape(1, self.nr_channels, self.img_size, self.img_size)
                    .to(kwargs.get("device", None))
                )
                .max(1)
                .indices
            )

            # Filter on samples that are classified correctly.
            if y_pred_perturbed == y:
                counts_corrs += 1

                # Append similarity score.
                similarity = self.similarity_func(a.flatten(), a_perturbed.flatten())
                if similarity < self.threshold:
                    counts_thres += 1

        self.last_results.append(float(counts_thres / counts_corrs))
        self.all_results.append(self.last_results)

        return self.last_results


if __name__ == "__main__":

    # Run tests!
    pass
