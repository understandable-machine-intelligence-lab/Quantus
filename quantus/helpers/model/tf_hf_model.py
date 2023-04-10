# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from functools import singledispatchmethod, cached_property, partial
from operator import contains
from typing import List, Generator, Dict

import keras
import numpy as np
import tensorflow as tf
from transformers import TFPreTrainedModel

from quantus.helpers.collection_utils import (
    value_or_default,
    filter_dict,
    add_default_items,
)
from quantus.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.tf_utils import (
    is_xla_compatible_platform,
    list_parameterizable_layers,
    random_layer_generator,
    supported_keras_engine_predict_kwargs,
)
from quantus.helpers.types import LayerOrderT


class TFHuggingFaceTextClassifier(TextClassifier, tf.Module):
    def __init__(
        self,
        model: TFPreTrainedModel,
        tokenizer: HuggingFaceTokenizer,
        softmax: bool | None = None,
        model_predict_kwargs: Dict[str, ...] | None = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model._jit_compile = is_xla_compatible_platform()
        model_predict_kwargs = filter_dict(
            value_or_default(model_predict_kwargs, lambda: {}),
            key_filter=partial(contains, supported_keras_engine_predict_kwargs()),
        )
        model_predict_kwargs = add_default_items(
            model_predict_kwargs, dict(batch_size=64)
        )
        self.model.predict = partial(self.model.predict, **model_predict_kwargs)

    def embedding_lookup(self, input_ids: np.ndarray | tf.Tensor) -> tf.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    @singledispatchmethod
    def predict(self, x_batch: List[str], **kwargs) -> np.ndarray:
        encoded_inputs = self.tokenizer.batch_encode(x_batch, return_tensors="tf")
        return self.model.predict(encoded_inputs, verbose=0, **kwargs).logits

    @predict.register(tf.Tensor)
    @predict.register(np.ndarray)
    def _(self, x_batch: tf.Tensor | np.ndarray, **kwargs) -> tf.Tensor:
        return self.model(None, inputs_embeds=x_batch, training=False, **kwargs).logits

    def get_random_layer_generator(
        self, order: LayerOrderT = "top_down", seed: int = 42
    ) -> Generator[TFHuggingFaceTextClassifier, None, None]:
        return random_layer_generator(self, order, seed, flatten_layers=True)

    @cached_property
    def random_layer_generator_length(self) -> int:
        return len(list_parameterizable_layers(self.get_model(), flatten_layers=True))

    def get_model(self) -> keras.Model:
        return self.model

    def state_dict(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def load_state_dict(self, original_parameters: List[np.ndarray]):
        self.model.set_weights(original_parameters)

    @singledispatchmethod
    def get_hidden_representations(
        self,
        x_batch: List[str],
        *args,
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

    @get_hidden_representations.register(tf.Tensor)
    @get_hidden_representations.register(np.ndarray)
    def _(
        self,
        x_batch: tf.Tensor | np.ndarray,
        *args,
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
