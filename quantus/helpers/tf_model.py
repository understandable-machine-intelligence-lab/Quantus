"""This model creates the ModelInterface for Tensorflow."""
from typing import Optional, Tuple, List

from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
import numpy as np
import tensorflow as tf
import gc

from ..helpers.model_interface import ModelInterface
from ..helpers import utils


class TensorFlowModel(ModelInterface):
    """Interface for tensorflow models."""

    def __init__(self, model, channel_first):
        super().__init__(model, channel_first)

    def predict(self, x, **kwargs):
        """Predict on the given input."""
        # Generally, one should always prefer keras predict to __call__
        # https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call

        softmax_act = kwargs.get("softmax", False)

        output_act = self.model.layers[-1].activation
        target_act = softmax if softmax_act else linear

        if output_act == target_act:
            return self.model.predict(x, verbose=0)

        config = self.model.layers[-1].get_config()
        config["activation"] = target_act

        weights = self.model.layers[-1].get_weights()

        output_layer = Dense(**config)(self.model.layers[-2].output)
        new_model = Model(inputs=[self.model.input], outputs=[output_layer])
        new_model.layers[-1].set_weights(weights)
        # we don't need TF to trace + compile this model. We're going to call it once only
        new_model.run_eagerly = True

        return new_model.predict(x, verbose=0)

    def shape_input(
        self,
        x: np.array,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batch: bool = False,
    ):
        """
        Reshape input into model expected input.
        channel_first: Explicitely state if x is formatted channel first (optional).
        """
        if channel_first is None:
            channel_first = utils.infer_channel_first
        if batch:
            x = x.reshape(x.shape[0], *shape)
        else:
            x = x.reshape(1, *shape)
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        return utils.make_channel_last(x, channel_first)

    def get_model(self):
        """Get the original torch/tf model."""
        return self.model

    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        return self.model.get_weights()

    def load_state_dict(self, original_parameters):
        """Set model's learnable parameters."""
        self.model.set_weights(original_parameters)

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        Set order to top_down for cascading randomization.
        Set order to independent for independent randomization.
        """
        original_parameters = self.state_dict()
        random_layer_model = clone_model(self.model)

        layers = [l for l in random_layer_model.layers if len(l.get_weights()) > 0]

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                random_layer_model.set_weights(original_parameters)
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, random_layer_model

    def get_hidden_layers_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> np.ndarray:

        if layer_indices is None:
            layer_indices = []
        if layer_names is None:
            layer_names = []

        def is_layer_of_interest(index, name):
            if layer_names == [] and layer_indices == []:
                return True
            return index in layer_indices or name in layer_names

        outputs_of_interest = []
        for i, layer in enumerate(self.model.layers):
            if is_layer_of_interest(i, layer.name):
                outputs_of_interest.append(layer.output)

        sub_model = tf.keras.Model(self.model.input, outputs_of_interest)
        # we don't need TF to trace + compile this model. We're going to call it once only
        sub_model.run_eagerly = True
        internal_representation = sub_model.predict(x, verbose=0)
        input_batch_size = x.shape[0]

        # Clean-up memory reserved for model's copy
        del sub_model
        gc.collect()

        if isinstance(internal_representation, np.ndarray):
            # If we requested outputs only of 1 layer, keras will already return np.ndarray
            return internal_representation.reshape((input_batch_size, -1))
        else:
            # otherwise, keras returns List of np.ndarray
            internal_representation = [
                i.reshape((input_batch_size, -1)) for i in internal_representation
            ]
            return np.hstack(internal_representation)
