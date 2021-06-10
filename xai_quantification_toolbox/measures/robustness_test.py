import numpy as np
from typing import Union
from .base import Measure
from ..helpers.similarity_functions import *
from ..helpers.perturbation_functions import *
from ..helpers.explanation_methods import *


class RobustnessTest(Measure):
    """
    Implements basis functionality for the following evaluation measures:

        • Continuity (Montavon et al., 2018)
        • Estimated Lipschitz constant (Alvarez-Melis, 2019)
        • Input independence (BAM)
        • avg-Sensitivity, max-Sensitivity (Yeh et al., 2019)
        • Monotonicity (Nguyen et al., 2020)

    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.perturbation_function = self.kwargs.get("perturbation_function", None)
        self.similarity_function = self.kwargs.get("similarity_function", None)

        super(RobustnessTest, self).__init__()

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        device,
        **kwargs
    ):
        assert (
            "xai_method" in kwargs
        ), "To run RobustnessTest specify 'xai_method' (str) e.g., 'Gradient'."
        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(device),
                x,
                y,
                xai_method=kwargs.get("xai_method", "Gradient"),
                device=device,
            )
            # model.attribute(batch=x, neuron_selection=y, xai_method=kwargs.get("xai_method", "gradient"),)

        results = []
        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            # Generate explanation based on perturbed input x.
            x_perturbed = self.perturbation_function(x.flatten())
            a_perturbed = explain(
                model.to(device),
                x_perturbed,
                y,
                xai_method=kwargs.get("xai_method", "Gradient"),
                device=device,
            )
            # model.attribute(batch=x_perturbed, neuron_selection=y, xai_method=kwargs.get("xai_method", "gradient"),)

            # Append similarity score.
            results.append(
                self.similarity_function(
                    a=a.flatten(),
                    b=a_perturbed.cpu().numpy().flatten(),
                    c=x.flatten(),
                    d=x_perturbed,
                )
            )

        return results


class ContinuityTest(RobustnessTest):
    """
    Continuity test by Montavon et al (2018)

    Explanation continuity quanitfies the strongest variation of the explanation in the input domain:
    ||R(x) - R(x')||_1 / ||x - x'||_2
    where R(x) is the explanation for input x and x' is the perturbed input.

    Current assumptions:
        • that input is squared
        • made an quantitative interpretation of visually determining how similar f(x) and R(x1) curves are...
        • using torch for explanation and model
    """

    def __init__(self, *args, **kwargs):
        """"""

        assert (
            kwargs.get("img_size", 224) % kwargs.get("nr_patches", 4) == 0
        ), "Set nr_patches so that the modulo remainder returns 0 given the image size."

        self.args = args
        self.kwargs = kwargs

        self.perturbation_function = self.kwargs.get(
            "perturbation_function", translation_x_direction
        )
        self.similarity_function = self.kwargs.get(
            "similarity_function", lipschitz_constant
        )

        self.img_size = kwargs.get("img_size", 224)
        self.nr_channels = kwargs.get("nr_channels", 3)

        self.nr_patches = kwargs.get("nr_patches", 4)
        self.patch_size = (self.img_size * 2) // self.nr_patches
        self.baseline_value = kwargs.get("baseline_value", 0.0)
        self.nr_steps = kwargs.get("nr_steps", 10)
        self.dx = self.img_size // self.nr_steps

        super(RobustnessTest, self).__init__()

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: Union[np.array, int],
        a_batch: Union[np.array, None],
        device,
        **kwargs
    ):
        assert (
            "xai_method" in kwargs
        ), "To run RobustnessTest specify 'xai_method' (str) e.g., 'Gradient'."
        assert (
            np.shape(x_batch)[0] == np.shape(a_batch)[0]
        ), "Inputs and attributions should include the same number of samples."

        if a_batch is None:
            explain(
                model.to(device),
                x,
                y,
                xai_method=kwargs.get("xai_method", "Gradient"),
                device=device,
            )
            # model.attribute(batch=x, neuron_selection=y, xai_method=kwargs.get("xai_method", "gradient"),)

        results = []  # {k: None for k in range(len(x_batch))}

        for ix, (x, y, a) in enumerate(zip(x_batch, y_batch, a_batch)):

            sub_results = {k: [] for k in range(self.nr_patches + 1)}

            for step in range(self.nr_steps):

                # Generate explanation based on perturbed input x.
                x_perturbed = self.perturbation_function(
                    x,
                    **{
                        "dx": (step + 1) * self.dx,
                        "baseline_value": self.baseline_value,
                    }
                )
                a_perturbed = explain(
                    model.to(device),
                    x_perturbed,
                    y,
                    xai_method=kwargs.get("xai_method", "Gradient"),
                    device=device,
                )
                # model.attribute(batch=x_perturbed, neuron_selection=y, xai_method=kwargs.get("xai_method", "gradient"),)

                # a_perturbed_test = np.ones_like(a_perturbed.cpu().numpy())
                # plt.imshow(np.moveaxis(x_perturbed.reshape(3, 224, 224), 0, 2))
                # plt.show()

                # Store the prediction score as the last element of the sub_results dictionary.
                y_pred = model(
                    torch.Tensor(x_perturbed)
                    .reshape(1, self.nr_channels, self.img_size, self.img_size)
                    .to(device)
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
                        self.similarity_function(
                            sub_results[self.nr_patches], sub_results[ix_patch]
                        )
                        for ix_patch in range(self.nr_patches)
                    ]
                )
            )  # results[ix] = sub_results

        return results
