from ..helpers.model_interface import ModelInterface
import torch
from ..helpers.utils import get_layers


class PyTorchModel(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, x, device):
        with torch.no_grad():
            pred = self.model(torch.Tensor(x).to(device))
        return pred.numpy()

    def shape_input(self, x, img_size, nr_channels):
        return x.reshape(1, nr_channels, img_size, img_size)

    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, original_parameters):
        self.model.load_state_dict(original_parameters)

    def get_layers(self, order):
        return get_layers(self.model, order=order)

    def randomize_layer(self, layer_name):
        layer = getattr(self.get_model(), layer_name)
        layer.reset_parameters()
