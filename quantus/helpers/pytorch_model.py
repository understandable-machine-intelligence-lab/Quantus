from copy import deepcopy

from ..helpers.model_interface import ModelInterface
import torch


class PyTorchModel(ModelInterface):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, x, softmax_act=False, **kwargs):
        device = kwargs.get("device", None)
        with torch.no_grad():
            pred = self.model(torch.Tensor(x).to(device))
        if softmax_act:
            return torch.nn.Softmax()(pred).numpy()
        return pred.numpy()

    def shape_input(self, x, img_size, nr_channels):
        return x.reshape(1, nr_channels, img_size, img_size)

    def get_model(self):
        return self.model

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, original_parameters):
        self.model.load_state_dict(original_parameters)

    def get_random_layer_generator(self, order: str = "top_down"):
        original_parameters = self.state_dict()
        random_layer_model = deepcopy(self.model)

        modules = [
            l
            for l in random_layer_model.named_modules()
            if (hasattr(l[1], "reset_parameters"))
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                random_layer_model.load_state_dict(original_parameters)
            module[1].reset_parameters()
            yield module[0], random_layer_model
