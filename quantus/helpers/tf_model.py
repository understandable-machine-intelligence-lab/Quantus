from ..helpers.model_interface import ModelInterface
from ..metrics import *
import numpy as np
#import tensorflow as tf
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
        # TODO: implement
        return None

    def load_state_dict(self, original_parameters):
        # TODO: implement
        return None

    def get_layers(self, order):
        # TODO: implement
        return None

    def randomize_layer(self, layer_name):
        # TODO: implement
        return None
