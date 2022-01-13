from ..helpers.model_interface import ModelInterface
from ..metrics import *
import numpy as np
from tensorflow.keras.activations import linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class TensorFlowModel(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, x, device=None):
        # Remove activation from last layer to get logits
        if self.model.layers[-1].activation != linear:
            config = self.model.layers[-1].get_config()
            weights = [x.numpy() for x in self.model.layers[-1].weights]

            config['activation'] = linear
            config['name'] = 'logits'

            output_layer = Dense(**config)(self.model.layers[-2].output)
            self.model = Model(inputs=[self.model.input], outputs=[output_layer])
            self.model.layers[-1].set_weights(weights)
        return self.model(x, training=False).numpy()

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
