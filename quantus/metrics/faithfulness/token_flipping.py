# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from typing import List, Optional, Literal, Callable

import numpy as np

from quantus.functions.loss_func import mse
from quantus.functions.normalise_func import normalize_sum_to_1
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.plotting import plot_token_flipping_experiment
from quantus.helpers.types import NormaliseFn, Explanation, Kwargs, ExplainFn, Any
from quantus.helpers.utils import safe_as_array, get_logits_for_labels
from quantus.metrics.base_batched import BatchedMetric

TaskT = Literal["pruning", "activation"]


class TokenFlipping(BatchedMetric):
    """
    References:
        - https://arxiv.org/abs/2202.07304
        - https://github.com/AmeenAli/XAI_Transformers/blob/main/SST/run_sst.py#L127
        - https://arxiv.org/pdf/2205.15389.pdf

    For the NLP experiments, we consider token sequences instead of graph nodes. The activation task starts with an
    empty sentence of “UNK” tokens, which are then gradually replaced with the original tokens in the order of highest
    to lowest relevancy. In the pruning task, we remove tokens from lowest to highest absolute relevance by replacing
    them with “UNK” tokens, similarly to the ablation experiments of Abnar & Zuidema (2020).
    """

    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[NormaliseFn] = normalize_sum_to_1,
            normalise_func_kwargs: Kwargs = None,
            return_aggregate: bool = False,
            aggregate_func=None,
            default_plot_func: Optional[Callable] = plot_token_flipping_experiment,
            disable_warnings: bool = False,
            display_progressbar: bool = False,
            mask_token: str = "[UNK]",
            task: TaskT = "pruning",
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
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        mask_token:
            Token which is used to mask inputs.
        task:
            Can be either pruning or activation.
        """
        super().__init__(
            abs,
            normalise,
            normalise_func,
            normalise_func_kwargs,
            return_aggregate,
            aggregate_func,
            default_plot_func,
            disable_warnings,
            display_progressbar,
            **kwargs,
        )
        if return_aggregate:
            raise ValueError(f"Token Flipping does not support aggregating instances.")
        if task not in ("pruning", "activation"):
            raise ValueError(f"Unsupported task, supported are: pruning, activation.")
        self.mask_token = mask_token
        self.task = task

    def __call__(
            self,
            model,
            x_batch: List[str],
            y_batch: Optional[np.ndarray],
            a_batch: Optional[List[Explanation]],
            explain_func: Optional[ExplainFn],
            explain_func_kwargs: Kwargs = None,
            model_predict_kwargs: Kwargs = None,
            softmax: Optional[bool] = None,
            s_batch: Optional[np.ndarray] = None,
            channel_first: Optional[bool] = None,
            device: Optional[str] = None,
            batch_size: int = 64,
            custom_batch: Optional[Any] = None,
            **kwargs,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        model:
            Torch or tensorflow model that is subject to explanation. Most probably, you will want to use
            `quantus.nlp.TorchHuggingFaceTextClassifier` or `quantus.nlp.TensorFlowHuggingFaceTextClassifier`,
            for out-of-the box support for models from Huggingface hub.
        x_batch:
            list, which contains the input data that are explained.
        y_batch:
            A np.ndarray which contains the output labels that are explained.
        a_batch:
            Pre-computed attributions i.e., explanations. Token and scores as well as scores only are supported.
        explain_func:
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        batch_size:
            Indicates size of batches, in which input dataset will be splitted.

        Returns
        -------

        score:
            MSE between original logits, and ones for masked inputs.

        """

        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=None,
            channel_first=None,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=None,
            device=device,
            batch_size=batch_size,
            custom_batch=None,
            **kwargs,
        )

    def evaluate_batch(
            self,
            model: TextClassifier,
            x_batch: List[str],
            y_batch: np.ndarray,
            a_batch: List[Explanation],
            s_batch=None,
            custom_batch=None,
    ) -> np.ndarray:
        if self.task == "pruning":
            scores = self._eval_pruning(model, x_batch, y_batch, a_batch)
        else:
            scores = self._eval_activation(model, x_batch, y_batch, a_batch)

        og_logits = get_logits_for_labels(model.predict(x_batch), y_batch)

        scores = mse(scores, og_logits, axis=1)

        # Move batch axis to 0's position.
        return scores.T

    def _eval_activation(
            self,
            model: TextClassifier,
            x_batch: List[str],
            y_batch: np.ndarray,
            a_batch: List[Explanation],
    ):
        batch_size = len(a_batch)
        num_tokens = len(a_batch[0][1])

        # We need to have tokens axis at positions 0, so we can just insert batches.
        scores = np.full(shape=(num_tokens, batch_size), fill_value=np.NINF)
        mask_indices_all = []
        for a in a_batch:
            # descending scores
            token_indices_sorted_by_scores = np.argsort(a[1])[::-1]
            mask_indices_all.append(token_indices_sorted_by_scores)

        mask_indices_all = np.asarray(mask_indices_all).T
        input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)
        input_ids = safe_as_array(input_ids)
        mask_token_id = model.tokenizer.token_id(self.mask_token)
        masked_input_ids = np.full_like(np.asarray(input_ids), fill_value=mask_token_id)

        for i, mask_indices_batch in enumerate(mask_indices_all):
            for index_in_batch, mask_index in enumerate(mask_indices_batch):
                masked_input_ids[index_in_batch][mask_index] = input_ids[
                    index_in_batch
                ][mask_index]

            embeddings = model.embedding_lookup(masked_input_ids)
            masked_logits = model(embeddings, **predict_kwargs)
            masked_logits = safe_as_array(masked_logits, force=True)
            masked_logits = get_logits_for_labels(masked_logits, y_batch)
            scores[i] = masked_logits

        return scores

    def _eval_pruning(
            self,
            model: TextClassifier,
            x_batch: List[str],
            y_batch: np.ndarray,
            a_batch: List[Explanation],
    ) -> np.ndarray:
        batch_size = len(a_batch)
        num_tokens = len(a_batch[0][1])

        # We need to have tokens axis at positions 0, so we can just insert batches.
        scores = np.full(shape=(num_tokens, batch_size), fill_value=np.NINF)
        mask_indices_all = []
        for a in a_batch:
            # ascending scores
            token_indices_sorted_by_scores = np.argsort(a[1])
            mask_indices_all.append(token_indices_sorted_by_scores)

        mask_indices_all = np.asarray(mask_indices_all).T
        input_ids, predict_kwargs = model.tokenizer.get_input_ids(x_batch)
        input_ids = safe_as_array(input_ids, force=True)
        mask_token_id = model.tokenizer.token_id(self.mask_token)

        for i, mask_indices_batch in enumerate(mask_indices_all):
            for index_in_batch, mask_index in enumerate(mask_indices_batch):
                input_ids[index_in_batch][mask_index] = mask_token_id

            embeddings = model.embedding_lookup(input_ids)
            masked_logits = model(embeddings, **predict_kwargs)
            masked_logits = safe_as_array(masked_logits, force=True)
            masked_logits = get_logits_for_labels(masked_logits, y_batch)
            scores[i] = masked_logits

        return scores

    @classmethod
    @property
    def data_domain_applicability(self):
        return ["NLP"]
