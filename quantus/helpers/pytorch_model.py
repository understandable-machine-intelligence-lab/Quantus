from ..helpers.model_interface import ModelInterface
import torch


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
