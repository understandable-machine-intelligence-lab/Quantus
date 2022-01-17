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

    def load_state_dict(self, parameters):
        self.model.load_state_dict(parameters)

    def get_layers(self, order: str = "top_down"):
        """Checks a pytorch model for randomizable layers and returns them in a dict."""
        layers = [
            module
            for module in self.model.named_modules()
            if hasattr(module[1], "reset_parameters")
        ]

        if order == "top_down":
            return layers[::-1]
        else:
            return layers

    def randomize_layer(self, layer_name):
        layer = getattr(self.get_model(), layer_name)
        layer.reset_parameters()
