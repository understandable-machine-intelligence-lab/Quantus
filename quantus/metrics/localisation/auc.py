"""This module contains the implementation of the AUC metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc

from ..base import Metric
from ...helpers import asserts
from ...helpers import warn_func
from ...helpers.model_interface import ModelInterface
from ...helpers.normalise_func import normalise_by_negative


class AUC(Metric):
    """
    Implementation of AUC metric by Fawcett et al., 2006.

    AUC is a ranking metric and  compares the ranking between attributions and a given ground-truth mask

    References:
        1) Fawcett, Tom. 'An introduction to ROC analysis' "Pattern Recognition Letters" Vol 27, Issue 8, 2006

    """

    @asserts.attributes_check
    def __init__(
        self,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
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
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_negative.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
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

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Asserts and warnings.
        if not self.disable_warnings:
            warn_func.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "ground truth mask i.e., the 's_batch' input as well as if "
                    "absolute values 'abs' are taken of the attributions "
                ),
                citation=(
                    "Fawcett, Tom. 'An introduction to ROC analysis' Pattern Recognition Letters"
                    " Vol 27, Issue 8, (2006)"
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray],
        s_batch: np.array,
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
                 Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
            explain_func: callable
                Callable generating attributions.
            explain_func_kwargs: dict, optional
                Keyword arguments to be passed to explain_func on call.
            model_predict_kwargs: dict, optional
                Keyword arguments to be passed to the model's predict method.
            softmax: boolean
                Indicates whether to use softmax probabilities or logits in model prediction. This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
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
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
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
         model (ModelInteface): A ModelInteface that is subject to explanation.
         x: np.ndarray
              The input to be evaluated on an instance-basis.
         y: np.ndarray
              The output to be evaluated on an instance-basis.
         a: np.ndarray
              The explanation to be evaluated on an instance-basis.
         a: np.ndarray
              The segmentation to be evaluated on an instance-basis.
         c: any
             The custom input to be evaluated on an instance-basis.
         p: any
             The custom preprocess input to be evaluated on an instance-basis.

         Returns
         -------
        : float
             The evaluation results.
        """
        # Return np.nan as result if segmentation map is empty.
        if np.sum(s) == 0:
            warn_func.warn_empty_segmentation()
            return np.nan

        # Prepare shapes.
        a = a.flatten()
        s = s.flatten().astype(bool)

        fpr, tpr, _ = roc_curve(y_true=s, y_score=a)
        score = auc(x=fpr, y=tpr)

        return score

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
                model (Union[torch.nn.Module, tf.keras.Model]): A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
                x_batch: np.ndarray
                A np.ndarray which contains the input data that are explained.
                y_batch: np.ndarray
                A np.ndarray which contains the output labels that are explained.
                a_batch: np.ndarray
                A Union[np.ndarray, None] which contains pre-computed attributions i.e., explanations.
                s_batch: np.ndarray
                A Union[np.ndarray, None] which contains segmentation masks that matches the input.
                custom_batch: any
                Gives flexibility ot the user to use for evaluation, can hold any variable.

            Returns
            -------
                (Tuple[ModelInterface, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any]):     In addition to the x_batch, y_batch, a_batch, s_batch and custom_batch,
        returning a custom preprocess batch (custom_preprocess_batch).

        """

        custom_preprocess_batch = [None for _ in x_batch]

        # Asserts.
        asserts.assert_segmentations(x_batch=x_batch, s_batch=s_batch)

        return (
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            custom_batch,
            custom_preprocess_batch,
        )
