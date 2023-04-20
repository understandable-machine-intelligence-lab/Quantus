from importlib import util
from typing import Generator, Any, List

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from transformers import TFPreTrainedModel, PreTrainedTokenizerBase

from quantus.helpers.collection_utils import value_or_default
from quantus.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.tf_utils import as_tensor
from quantus.helpers.types import LayerOrderT

if util.find_spec("transformers_gradients"):
    from transformers_gradients.model_utils import (
        convert_graph_to_tensor_rt,
        build_embeddings_model,
    )
    from transformers_gradients.functions import bounding_shape
    from transformers_gradients.types import UserObject
    from transformers_gradients.util import as_tensor


else:
    UserObject = type(None)


# Turns out TensorRT engine does not record gradients
# However it still can be used with LIME and SHAP or with saved_model format


class TensorRTModel(TextClassifier):
    tokenizer: HuggingFaceTokenizer
    input_ids_model: UserObject
    embeddings_model: UserObject

    def __init__(
        self,
        hf_model: TFPreTrainedModel,
        tokenizer: HuggingFaceTokenizer,
        fallback_to_saved_model: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model_family = hf_model.base_model_prefix
        self.input_ids_model = convert_graph_to_tensor_rt(
            hf_model, fallback_to_saved_model
        )
        embeddings_model = build_embeddings_model(hf_model)
        self.embeddings_model = convert_graph_to_tensor_rt(
            embeddings_model, fallback_to_saved_model
        )

    def embedding_lookup(self, input_ids: tf.Tensor) -> tf.Tensor:
        return tf.gather(
            getattr(self.input_ids_model, self.model_family).embeddings.weight,
            input_ids,
        )

    def predict(
        self, x_batch: List[str] | tf.Tensor, attention_mask=None, **kwargs
    ) -> tf.Tensor:
        if isinstance(x_batch, (tf.Tensor, np.ndarray)):
            x_batch = as_tensor(x_batch)
            attention_mask = value_or_default(
                attention_mask,
                lambda: tf.ones(bounding_shape(input_ids), dtype=tf.int32),
            )
            attention_mask = tf.cast(attention_mask, dtype=tf.int32)
            return self.embeddings_model.signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY](
                inputs_embeds=as_tensor(x_batch), attention_mask=attention_mask
            )["classifier"]

        # Convert to vocabulary ids
        input_ids, predict_kwargs = self.tokenizer.get_input_ids(x_batch)
        attention_mask = predict_kwargs.get("attention_mask")
        attention_mask = value_or_default(
            attention_mask, lambda: tf.ones(bounding_shape(input_ids), dtype=tf.int32)
        )
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        return self.input_ids_model.signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY](
            input_ids=input_ids, attention_mask=attention_mask
        )["logits"]

    def get_random_layer_generator(
        self, order: LayerOrderT = "top_down", seed: int = 42
    ) -> Generator[Any, None, None]:
        parameters = self.input_ids_model.variables.copy()
        parameters_copy = self.input_ids_model.variables.copy()

        np.random.seed(seed)

        variable_indexes = tf.range(len(parameters))

        if order == "top_down":
            variable_indexes = variable_indexes[::-1]

        for i in variable_indexes:
            if order == "independent":
                self.load_state_dict(parameters_copy)

            parameters[i] = tf.Variable(
                tf.random.experimental.stateless_shuffle(
                    parameters[i], seed=[seed, seed]
                ),
                name=parameters[i].name,
            )
            self.load_state_dict(parameters)
            yield parameters[i].name, self
        # Restore original weights.
        self.load_state_dict(parameters_copy)

    @property
    def random_layer_generator_length(self) -> int:
        return len(self.input_ids_model.variables.copy())

    def state_dict(self) -> List[tf.Variable]:
        return self.input_ids_model.variables

    def load_state_dict(self, original_parameters: List[tf.Variable]):
        self.input_ids_model.variables = original_parameters
        self.embeddings_model.variable = original_parameters

    def get_model(self):
        raise ValueError("Not supported")

    def get_hidden_representations(self, x, *args, **kwargs):
        raise ValueError("Not supported")
