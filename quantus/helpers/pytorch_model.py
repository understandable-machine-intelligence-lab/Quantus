"""This model creates the ModelInterface for PyTorch."""
from contextlib import suppress
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List
import torch
import numpy as np

from ..helpers.model_interface import ModelInterface
from ..helpers import utils


class PyTorchModel(ModelInterface):
    """Interface for torch models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        device: Optional[str] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            predict_kwargs=predict_kwargs,
        )
        self.device = device

    def predict(self, x: np.ndarray, grad: bool = False, **kwargs):
        """Predict on the given input."""

        # Use kwargs of predict call if specified, but don't overwrite object attribute
        predict_kwargs = {**self.predict_kwargs, **kwargs}

        if self.model.training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")

        grad_context = torch.no_grad() if not grad else suppress()

        with grad_context:
            pred = self.model(torch.Tensor(x).to(self.device), **predict_kwargs)
            if self.softmax:
                pred = torch.nn.Softmax(dim=-1)(pred)
            if pred.requires_grad:
                return pred.detach().cpu().numpy()
            return pred.cpu().numpy()

    def shape_input(
        self,
        x: np.array,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ):
        """
        Reshape input into model expected input.
        channel_first: Explicitely state if x is formatted channel first (optional).
        """
        if channel_first is None:
            channel_first = utils.infer_channel_first(x)

        # Expand first dimension if this is just a single instance.
        if not batched:
            x = x.reshape(1, *shape)

        # Set channel order according to expected input of model.
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        raise ValueError("Channel first order expected for a torch model.")

    def get_model(self):
        """Get the original torch/tf model."""
        return self.model

    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        return self.model.state_dict()

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
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
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()
            yield module[0], random_layer_model

    def get_hidden_layers_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> np.ndarray:

        device = kwargs.get("device", torch.device("cpu"))

        if layer_indices is None:
            layer_indices = []
        if layer_names is None:
            layer_names = []

        def is_layer_of_interest(index, name):
            if layer_names == [] and layer_indices == []:
                return True
            return index in layer_indices or name in layer_names

        # skip last layer
        hidden_layers = [layer for layer in self.model.named_modules()][:-1]
        # skip modules defined by subclassing API
        hidden_layers = list(
            filter(
                lambda l: not isinstance(
                    l[1], (self.model.__class__, torch.nn.Sequential)
                ),
                hidden_layers,
            )
        )

        batch_size = x.shape[0]
        hidden_outputs = []

        # We register forward hook on layers of interest,
        # which just saves the flattened layers' outputs to list,
        # Then we execute forward pass, and stack them in 2D tensor

        def hook(module, module_in, module_out):
            arr = module_out.cpu().numpy()
            arr = arr.reshape((batch_size, -1))
            hidden_outputs.append(arr)

        new_hooks = []
        # Save handles of registered hooks, so we can clean them up later
        for index, (name, layer) in enumerate(hidden_layers):
            if is_layer_of_interest(index, name):
                handle = layer.register_forward_hook(hook)
                new_hooks.append(handle)

        # Execute forward pass
        with torch.no_grad():
            self.model(torch.Tensor(x).to(device))

        # Cleanup
        [i.remove() for i in new_hooks]

        return np.hstack(hidden_outputs)
