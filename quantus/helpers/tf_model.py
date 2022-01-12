from ..helpers.model_interface import ModelInterface
from ..metrics import *
import numpy as np


class TensorFlowModel(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, x, device=None):
        return self.model(x, training=False).numpy()

    def shape_input(self, x, img_size, nr_channels):
        x = x.reshape(1, nr_channels, img_size, img_size)
        return np.moveaxis(x, 1, -1)

    def get_model(self):
        return self.model
