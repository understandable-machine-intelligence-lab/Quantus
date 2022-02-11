"""This model creates the ModelInterface for PyTorch."""
import torch
import numpy as np
from copy import deepcopy
from contextlib import suppress
from ..helpers.model_interface import ModelInterface


class PyTorchModel(ModelInterface):
    """Interface for torch models."""

    def __init__(self, model, channel_first):
        super().__init__(model, channel_first)

    def predict(self, x, softmax_act=False, **kwargs):
        """Predict on the given input."""
        if self.model.training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")

        device = kwargs.get("device", None)
        grad = kwargs.get("grad", False)
        grad_context = torch.no_grad() if not grad else suppress()

        with grad_context:
            pred = self.model(torch.Tensor(x).to(device))
            if softmax_act:
                pred = torch.nn.Softmax()(pred)
            if pred.requires_grad:
                return pred.detach().cpu().numpy()
            return pred.cpu().numpy()

    def shape_input(self, x, img_size, nr_channels):
        """Reshape input into model expected input."""
        x = x.reshape(1, nr_channels, img_size, img_size)
        if self.channel_first:
            return x
        raise ValueError("Channel first order expected for a torch model.")

    def get_model(self):
        """Get the original torch/tf model."""
        return self.model

    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        return self.model.state_dict()

    def get_random_layer_generator(self, order: str = "top_down"):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        Set order to top_down for cascading randomization.
        Set order to independent for independent randomization.
        """
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
