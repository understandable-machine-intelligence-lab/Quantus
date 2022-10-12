"""This module contains the implementation of the Infidelity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base import PerturbationMetric
from ...helpers import warn_func
from ...helpers import asserts
from ...helpers import utils
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative
from ...helpers.perturb_func import baseline_replacement_by_indices
from ...helpers.loss_func import mse


class Infidelity(PerturbationMetric):
    """
    Implementation of Infidelity by Yeh et al., 2019.

    Explanation infidelity represents the expected mean square error
    between 1) a dot product of an attribution and input perturbation and
    2) difference in model output after significant perturbation.

    Assumptions:
        - The original implementation (https://github.com/chihkuanyeh/saliency_evaluation/
        blob/master/infid_sen_utils.py) supports perturbation of Gaussian noise and squared patches.
        In this implementation, we use squared patches as the default option.
        - Since we use squared patches in this implementation, the metric is only applicable
        to 3-dimensional (image) data. To extend the applicablity to other data domains, adjustments
        to the current implementation might be necessary.

    References:
        1) Chih-Kuan Yeh, Cheng-Yu Hsieh, and Arun Sai Suggala.
        "On the (In)fidelity and Sensitivity of Explanations."
        33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
    """

    @asserts.attributes_check
    def __init__(
        self,
        loss_func: str = "mse",
        perturb_patch_sizes: List[int] = None,
        n_perturb_samples: int = 10,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        loss_func: string
            Loss function, default="mse".
        perturb_patch_sizes: list
            List of patch sizes to be perturbed. If None, the defaul is used, default=[4].
        features_in_step: integer
            The size of the step, default=1.
        n_perturb_samples: integer
            The number of samples to be perturbed, default=10.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=False.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=False.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        if normalise_func is None:
            normalise_func = normalise_by_negative

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save metric-specific attributes.
        if isinstance(loss_func, str):
            if loss_func == "mse":
                loss_func = mse
            else:
                raise ValueError(f"loss_func must be in ['mse'] but is: {loss_func}")
        self.loss_func = loss_func

        if perturb_patch_sizes is None:
            perturb_patch_sizes = [4]
        self.perturb_patch_sizes = perturb_patch_sizes
        self.n_perturb_samples = n_perturb_samples
        self.nr_channels = None

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', perturbation function 'perturb_func',"
                    "number of perturbed samples 'n_perturb_samples', the loss function 'loss_func' "
                    "aggregation boolean 'aggregate'"
                ),
                data_domain_applicability=(
                    f"Also, the current implementation only works for 3-dimensional (image) data."
                ),
                citation=(
                    "Chih-Kuan, Yeh, et al. 'On the (In)fidelity and Sensitivity of Explanations'"
                    "arXiv:1901.09392 (2019)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        custom_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict[str, Any]] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        softmax: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        custom_batch: any
            Any object that can be passed to the evaluation process.
            Gives flexibility to the user to adapt for implementing their own metric.
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        last_results: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=custom_batch,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
            **kwargs,
        )

    def evaluate_instance(
        self,
        i: int,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
        c: Any,
        p: Any,
    ) -> float:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
        i: integer
            The evaluation instance.
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x: np.ndarray
            The input to be evaluated on an instance-basis.
        y: np.ndarray
            The output to be evaluated on an instance-basis.
        a: np.ndarray
            The explanation to be evaluated on an instance-basis.
        s: np.ndarray
            The segmentation to be evaluated on an instance-basis.
        c: any
            The custom input to be evaluated on an instance-basis.
        p: any
            The custom preprocess input to be evaluated on an instance-basis.

        Returns
        -------
        float
            The evaluation results.
        """

        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        results = []

        for _ in range(self.n_perturb_samples):

            sub_results = []

            for patch_size in self.perturb_patch_sizes:

                pred_deltas = np.zeros(
                    (int(a.shape[1] / patch_size), int(a.shape[2] / patch_size))
                )
                a_sums = np.zeros(
                    (int(a.shape[1] / patch_size), int(a.shape[2] / patch_size))
                )
                x_perturbed = x.copy()
                pad_width = patch_size - 1

                for i_x, top_left_x in enumerate(range(0, x.shape[1], patch_size)):

                    for i_y, top_left_y in enumerate(range(0, x.shape[2], patch_size)):

                        # Perturb input patch-wise.
                        x_perturbed_pad = utils._pad_array(
                            x_perturbed, pad_width, mode="edge", padded_axes=self.a_axes
                        )
                        patch_slice = utils.create_patch_slice(
                            patch_size=patch_size,
                            coords=[top_left_x, top_left_y],
                        )

                        x_perturbed_pad = self.perturb_func(
                            arr=x_perturbed_pad,
                            indices=patch_slice,
                            indexed_axes=self.a_axes,
                            **self.perturb_func_kwargs,
                        )

                        # Remove padding.
                        x_perturbed = utils._unpad_array(
                            x_perturbed_pad, pad_width, padded_axes=self.a_axes
                        )

                        # Predict on perturbed input x_perturbed.
                        x_input = model.shape_input(
                            x_perturbed, x.shape, channel_first=True
                        )
                        warn_func.warn_perturbation_caused_no_change(
                            x=x, x_perturbed=x_input
                        )
                        y_pred_perturb = float(model.predict(x_input)[:, y])

                        x_diff = x - x_perturbed
                        a_diff = np.dot(
                            np.repeat(a, repeats=self.nr_channels, axis=0), x_diff
                        )

                        pred_deltas[i_x][i_y] = y_pred - y_pred_perturb
                        a_sums[i_x][i_y] = np.sum(a_diff)

                sub_results.append(
                    self.loss_func(a=pred_deltas.flatten(), b=a_sums.flatten())
                )

            results.append(np.mean(sub_results))

        return np.mean(results)

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> Tuple[
        ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any
    ]:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        tuple
            In addition to the x_batch, y_batch, a_batch, s_batch and custom_batch,
            returning a custom preprocess batch (custom_preprocess_batch).
        """

        custom_preprocess_batch = [None for _ in x_batch]

        # Infer number of input channels.
        self.nr_channels = x_batch.shape[1]

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
