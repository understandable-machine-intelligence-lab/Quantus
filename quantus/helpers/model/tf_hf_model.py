# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatchmethod
from typing import List, Generator, Optional

import numpy as np
import tensorflow as tf
from transformers import (
    TFPreTrainedModel,
)

from quantus.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.helpers.model.model_interface import (
    HiddenRepresentationsModel,
)
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.model.tf_model import TFModelRandomizer
from quantus.helpers.tf_utils import is_xla_compatible_model




class TFHuggingFaceTextClassifier(
    TextClassifier,
    TFNestedModelRandomizer,
    HiddenRepresentationsModel,
):
    def __init__(self, model: TFPreTrainedModel, softmax: bool = None):
        handle = model.config._name_or_path  # noqa
        self._tokenizer = HuggingFaceTokenizer(handle)
        self.model = model
        self.model._jit_compile = is_xla_compatible_model(self.model)

    def embedding_lookup(self, input_ids) -> tf.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def __call__(self, inputs_embeds: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.model(
            None, inputs_embeds=inputs_embeds, training=False, **kwargs
        ).logits

    def predict(self, text: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self.tokenizer.batch_encode(text, return_tensors="tf")
        return self.model.predict(encoded_inputs, verbose=0, **kwargs).logits

    @singledispatchmethod
    def get_hidden_representations(
        self,
        x_batch,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> np.ndarray:
        encoded_batch = self.tokenizer.batch_encode(x_batch)
        hidden_states = self.model(
            **encoded_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = tf.transpose(tf.stack(hidden_states), [1, 0, 2, 3])
        return hidden_states

    @get_hidden_representations.register
    def _(
        self,
        x_batch: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return self.get_hidden_representations(tf.constant(x_batch), *args, **kwargs)

    @get_hidden_representations.register
    def _(
        self,
        x_batch: tf.Tensor,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> np.ndarray:
        hidden_states = self.model(
            None,
            inputs_embeds=x_batch,
            training=False,
            output_hidden_states=True,
            **kwargs,
        ).hidden_states
        hidden_states = tf.transpose(tf.stack(hidden_states), [1, 0, 2, 3])
        return hidden_states

    @property
    def tokenizer(self):
        return self._tokenizer
