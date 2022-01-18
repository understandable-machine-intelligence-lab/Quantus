from ..helpers.model_interface import ModelInterface
from ..metrics import *
import numpy as np
import io
import h5py

from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.activations import linear, softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class TensorFlowModel(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, x, softmax_act=False, **kwargs):
        output_act = self.model.layers[-1].activation
        target_act = softmax if softmax_act else linear

        if output_act == target_act:
            return self.model(x, training=False).numpy()

        config = self.model.layers[-1].get_config()
        config['activation'] = target_act

        weights = [x.numpy() for x in self.model.layers[-1].weights]

        output_layer = Dense(**config)(self.model.layers[-2].output)
        new_model = Model(inputs=[self.model.input], outputs=[output_layer])
        new_model.layers[-1].set_weights(weights)

        return new_model(x, training=False).numpy()

    def shape_input(self, x, img_size, nr_channels):
        x = x.reshape(1, nr_channels, img_size, img_size)
        return np.moveaxis(x, 1, -1)

    def get_model(self):
        return self.model

    def state_dict(self):
        # TODO: research less questionable approach
        bytes_file = io.BytesIO()
        with h5py.File(bytes_file, 'w') as f:
            hdf5_format.save_weights_to_hdf5_group(f, self.model.layers)
        return bytes_file

    def load_state_dict(self, original_parameters):
        with h5py.File(original_parameters, 'r') as f:
            hdf5_format.load_weights_from_hdf5_group(f, self.model.layers)

    def get_random_layer_generator(self, order: str = "top_down"):
        original_parameters = self.state_dict()

        layers = [
            layer
            for layer in self.model.layers
            if (hasattr(layer, 'kernel_initializer') or
                hasattr(layer, 'bias_initializer'))
        ]

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                self.load_state_dict(original_parameters)
            weights = layer.get_weights()
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, self.model

        self.load_state_dict(original_parameters)
