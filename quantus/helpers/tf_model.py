"""This model creates the ModelInterface for Tensorflow."""
from typing import Optional, Tuple

from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
import numpy as np

from ..helpers.model_interface import ModelInterface
from ..helpers import utils


class TensorFlowModel(ModelInterface):
    """Interface for tensorflow models."""

    def __init__(self, model, channel_first):
        super().__init__(model, channel_first)

    def predict(self, x, **kwargs):
        """Predict on the given input."""

        softmax_act = kwargs.get("softmax", False)

        output_act = self.model.layers[-1].activation
        target_act = softmax if softmax_act else linear

        if output_act == target_act:
            return self.model(x, training=False).numpy()

        config = self.model.layers[-1].get_config()
        config["activation"] = target_act

        weights = self.model.layers[-1].get_weights()

        output_layer = Dense(**config)(self.model.layers[-2].output)
        new_model = Model(inputs=[self.model.input], outputs=[output_layer])
        new_model.layers[-1].set_weights(weights)

        return new_model(x, training=False).numpy()

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
