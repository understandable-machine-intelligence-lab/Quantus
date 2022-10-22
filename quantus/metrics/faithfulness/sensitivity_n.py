"""This module contains the implementation of the Sensitivity-N metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantus.helpers import asserts
from quantus.helpers import plotting
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.perturb_func import baseline_replacement_by_indices
from quantus.functions.similarity_func import correlation_pearson
from quantus.metrics.base import PerturbationMetric


class SensitivityN(PerturbationMetric):
    """
    Implementation of Sensitivity-N test by Ancona et al., 2019.

    An attribution method satisfies Sensitivity-n when the sum of the attributions for any subset of features of
    cardinality n is equal to the variation of the output Sc caused removing the features in the subset. The test
    computes the correlation between sum of attributions and delta output.

    Pearson correlation coefficient (PCC) is computed between the sum of the attributions and the variation in the
    target output varying n from one to about 80% of the total number of features, where an average across a thousand
    of samples is reported. Sampling is performed using a uniform probability distribution over the features.

    References:
        1) Marco Ancona et al.: "Towards better understanding of gradient-based attribution
        methods for deep neural networks." ICLR (Poster) (2018).

    """

    @asserts.attributes_check
    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        n_max_percentage: float = 0.8,
        features_in_step: int = 1,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Callable = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        similarity_func: callable
            Similarity function applied to compare input and perturbed input,
            default=correlation_pearson.
        n_max_percentage: float
            The percentage of features to iteratively evaluatede, fault=0.8.
        features_in_step: integer
            The size of the step, default=1.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
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
            normalise_func = normalise_by_max

        if perturb_func is None:
            perturb_func = baseline_replacement_by_indices
        perturb_func = perturb_func

        if perturb_func_kwargs is None:
            perturb_func_kwargs = {}
        perturb_func_kwargs["perturb_baseline"] = perturb_baseline

        if default_plot_func is None:
            default_plot_func = plotting.plot_sensitivity_n_experiment

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
        if similarity_func is None:
            similarity_func = correlation_pearson
        self.similarity_func = similarity_func
        self.n_max_percentage = n_max_percentage
        self.features_in_step = features_in_step

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the patch size for masking "
                    "'patch_size', similarity function 'similarity_func' and the number "
                    "of features to iteratively evaluate 'n_max_percentage'"
                ),
                citation=(
                    "Ancona, Marco, et al. 'Towards better understanding of gradient-based "
                    "attribution methods for deep neural networks.' arXiv preprint "
                    "arXiv:1711.06104 (2017)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
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
        model: torch.nn.Module, tf.keras.Model
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
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        s: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Evaluate instance gets model and data for a single instance as input and returns the evaluation result.

        Parameters
        ----------
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

        Returns
        -------
            (Dict[str, List[float]]): The evaluation results.
        """

        # Reshape the attributions.
        a = a.flatten()

        # Get indices of sorted attributions (descending).
        a_indices = np.argsort(-a)

        # Predict on x.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        att_sums = []
        pred_deltas = []
        x_perturbed = x.copy()

        for i_ix, a_ix in enumerate(a_indices[:: self.features_in_step]):

            # Perturb input by indices of attributions.
            a_ix = a_indices[
                (self.features_in_step * i_ix) : (self.features_in_step * (i_ix + 1))
            ]
            x_perturbed = self.perturb_func(
                arr=x_perturbed,
                indices=a_ix,
                indexed_axes=self.a_axes,
                **self.perturb_func_kwargs,
            )
            warn.warn_perturbation_caused_no_change(x=x, x_perturbed=x_perturbed)

            # Sum attributions.
            att_sums.append(float(a[a_ix].sum()))

            x_input = model.shape_input(x_perturbed, x.shape, channel_first=True)
            y_pred_perturb = float(model.predict(x_input)[:, y])
            pred_deltas.append(y_pred - y_pred_perturb)

        # Each list-element of self.last_results will be such a dictionary
        # We will unpack that later in custom_postprocess().
        return {"att_sums": att_sums, "pred_deltas": pred_deltas}

    def custom_preprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        custom_batch: Optional[np.ndarray],
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
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
        None
        """
        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

    def custom_postprocess(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Post-process the evaluation results.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.

        Returns
        -------
        None
        """
        max_features = int(
            self.n_max_percentage * np.prod(x_batch.shape[2:]) // self.features_in_step
        )

        # Get pred_deltas and att_sums from result list.
        sub_results_pred_deltas: List[Any] = [
            r["pred_deltas"] for r in self.last_results
        ]
        sub_results_att_sums: List[Any] = [r["att_sums"] for r in self.last_results]

        # Re-arrange sub-lists so that they are sorted by n.
        sub_results_pred_deltas_l: Dict[int, Any] = {k: [] for k in range(max_features)}
        sub_results_att_sums_l: Dict[int, Any] = {k: [] for k in range(max_features)}

        for k in range(max_features):
            for pred_deltas_instance in sub_results_pred_deltas:
                sub_results_pred_deltas_l[k].append(pred_deltas_instance[k])
            for att_sums_instance in sub_results_att_sums:
                sub_results_att_sums_l[k].append(att_sums_instance[k])

        # Compute the similarity for each n.
        self.last_results = [
            self.similarity_func(
                a=sub_results_att_sums_l[k], b=sub_results_pred_deltas_l[k]
            )
            for k in range(max_features)
        ]
