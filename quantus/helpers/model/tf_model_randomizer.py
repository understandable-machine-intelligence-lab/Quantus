import tensorflow as tf
import numpy as np

from quantus.helpers.model.model_interface import RandomisableModel
from quantus.helpers.model.tf_model import TFModelWrapper


class TFModelRandomizer(RandomisableModel, TFModelWrapper):
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42, flatten_layers=False
    ):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

        Parameters
        ----------
        order: string
            The various ways that a model's weights of a layer can be randomised.
        seed: integer
            The seed of the random layer generator.
        flatten_layers:
            If set to true, will flatten nested layers.


        Returns
        -------
        generator:
            Generator, which in each iteration yields the layer name and the model.
            After generator is closed, original parameters are restored.


        """
        original_parameters = self.state_dict().copy()
        layers = self.list_parameterizable_layers(self.model, flatten_layers)

        np.random.seed(seed)

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                self.load_state_dict(original_parameters)

            weights = layer.get_weights()
            layer.set_weights(tf.nest.map_structure(np.random.permutation, weights))
            yield layer.name, self

        # Restore original weights.
        self.load_state_dict(original_parameters)

    @property
    def random_layer_generator_length(self) -> int:
        return len(self.list_parameterizable_layers(self.model))


class TFNestedModelRandomizer(TFModelRandomizer):
    @property
    def random_layer_generator_length(self) -> int:
        return len(self.list_parameterizable_layers(self.model, flatten_layers=True))

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42, **kwargs
    ):
        return super().get_random_layer_generator(order, seed, flatten_layers=True)
