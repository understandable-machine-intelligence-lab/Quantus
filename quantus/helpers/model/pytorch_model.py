"""This model creates the ModelInterface for PyTorch."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from contextlib import suppress
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import torch

from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface


class PyTorchModel(ModelInterface):
    """Interface for torch models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialisation of PyTorchModel class.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A model to be wrapped in the ModelInterface.
        channel_first: boolean, optional
             Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        """
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            model_predict_kwargs=model_predict_kwargs,
        )
        self.device = device

    def predict(self, x: np.ndarray, grad: bool = False, **kwargs) -> np.array:
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray
            A given input that the wrapped model predicts on.
        grad: boolean
            Indicates if gradient-calculation is disabled or not.
        kwargs: optional
            Keyword arguments.

        Returns
        --------
        np.ndarray
            predictions of the same dimension and shape as the input, values in the range [0, 1].
        """

        # Use kwargs of predict call if specified, but don't overwrite object attribute
        model_predict_kwargs = {**self.model_predict_kwargs, **kwargs}

        if self.model.training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")

        grad_context = torch.no_grad() if not grad else suppress()

        with grad_context:
            pred = self.model(torch.Tensor(x).to(self.device), **model_predict_kwargs)
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
    ) -> np.array:
        """
        Reshape input into model expected input.

        Parameters
        ----------
        x: np.ndarray
             A given input that is shaped.
        shape: Tuple[int...]
            The shape of the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        batched: boolean
            Indicates if the first dimension should be expanded or not, if it is just a single instance.

        Returns
        -------
        np.ndarray
            A reshaped input.
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

    def get_model(self) -> torch.nn:
        """
        Get the original torch model.
        """
        return self.model

    def state_dict(self) -> dict:
        """
        Get a dictionary of the model's learnable parameters.
        """
        return self.model.state_dict()

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

        Parameters
        ----------
        order: string
            The various ways that a model's weights of a layer can be randomised.
        seed: integer
            The seed of the random layer generator.

        Returns
        -------
        layer.name, random_layer_model: string, torch.nn
            The layer name and the model.
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

    def sample(
        self,
        mean: float,
        std: float,
        noise_type: str = "multiplicative",
    ) -> torch.nn:
        """
        Sample a model by means of adding normally distributed noise.

        Parameters
        ----------
        mean: float
            The mean point to sample from.
        std: float
            The standard deviation to sample from.
        noise_type: string
            Noise type could be either 'additive' or 'multiplicative'.

        Returns
        -------
        model_copy: torch.nn
            A noisy copy of the orginal model.
        """

        distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
        original_parameters = self.state_dict()
        model_copy = deepcopy(self.model)
        model_copy.load_state_dict(original_parameters)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not std == 0.0:
            with torch.no_grad():
                for layer in model_copy.parameters():
                    if noise_type == "additive":
                        layer.add_(distribution.sample(layer.size()).to(layer.device))
                    elif noise_type == "multiplicative":
                        layer.mul_(distribution.sample(layer.size()).to(layer.device))
                    else:
                        raise ValueError(
                            "Set noise_type to either 'multiplicative' "
                            "or 'additive' (string) when you sample the model."
                        )
        return model_copy

    def add_mean_shift_to_first_layer(
        self,
        input_shift: Union[int, float],
        shape: tuple,
    ):
        """
        Consider the first layer neuron before non-linearity: z = w^T * x1 + b1. We update
        the bias b1 to b2:= b1 - w^T * m (= 2*b1 - (w^T * m + b1)). The operation is necessary
        for Input Invariance metric.


        Parameters
        ----------
        input_shift: Union[int, float]
            Shift to be applied.
        shape: tuple
            Model input shape, ndim = 4.

        Returns
        -------
        random_layer_model: torch.nn
            The resulting model with a shifted first layer.
        """
        with torch.no_grad():

            new_model = deepcopy(self.model)

            modules = [l for l in new_model.named_modules()]
            module = modules[1]

            delta = torch.zeros(size=shape).fill_(input_shift)
            fw = module[1].forward(delta)[0]

            for i in range(module[1].out_channels):
                if self.channel_first:
                    module[1].bias[i] = torch.nn.Parameter(
                        2 * module[1].bias[i] - torch.unique(fw[i])[0]
                    )
                else:
                    module[1].bias[i] = torch.nn.Parameter(
                        2 * module[1].bias[i] - torch.unique(fw[..., i])[0]
                    )

        return new_model

    def get_hidden_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:

        """
        Compute the model's internal representation of input x.
        In practice, this means, executing a forward pass and then, capturing the output of layers (of interest).
        As the exact definition of "internal model representation" is left out in the original paper (see: https://arxiv.org/pdf/2203.06877.pdf),
        we make the implementation flexible.
        It is up to the user whether all layers are used, or specific ones should be selected.
        The user can therefore select a layer by providing 'layer_names' (exclusive) or 'layer_indices'.

        Parameters
        ----------
        x: np.ndarray
            4D tensor, a batch of input datapoints
        layer_names: List[str]
            List with names of layers, from which output should be captured.
        layer_indices: List[int]
            List with indices of layers, from which output should be captured.
            Intended to use in case, when layer names are not unique, or unknown.

        Returns
        -------
        L: np.ndarray
            2D tensor with shape (batch_size, None)
        """

        device = self.device if self.device is not None else "cpu"
        all_layers = [*self.model.named_modules()]
        num_layers = len(all_layers)

        if layer_indices is None:
            layer_indices = []

        # E.g., user can provide index -1, in order to get only representations of the last layer.
        # E.g., for 7 layers in total, this would correspond to positive index 6.
        positive_layer_indices = [
            i if i >= 0 else num_layers + i for i in layer_indices
        ]

        if layer_names is None:
            layer_names = []

        def is_layer_of_interest(layer_index: int, layer_name: str):
            if layer_names == [] and positive_layer_indices == []:
                return True
            return layer_index in positive_layer_indices or layer_name in layer_names

        # skip modules defined by subclassing API.
        hidden_layers = list(  # type: ignore
            filter(
                lambda l: not isinstance(
                    l[1], (self.model.__class__, torch.nn.Sequential)
                ),
                all_layers,
            )
        )

        batch_size = x.shape[0]
        hidden_outputs = []

        # We register forward hook on layers of interest, which just saves the flattened layers' outputs to list.
        # Then we execute forward pass and stack them in 2D tensor.
        def hook(module, module_in, module_out):
            arr = module_out.cpu().numpy()
            arr = arr.reshape((batch_size, -1))
            hidden_outputs.append(arr)

        new_hooks = []
        # Save handles of registered hooks, so we can clean them up later.
        for index, (name, layer) in enumerate(hidden_layers):
            if is_layer_of_interest(index, name):
                handle = layer.register_forward_hook(hook)
                new_hooks.append(handle)

        if len(new_hooks) == 0:
            raise ValueError("No hidden representations were selected.")

        # Execute forward pass.
        with torch.no_grad():
            self.model(torch.Tensor(x).to(device))

        # Cleanup.
        [i.remove() for i in new_hooks]
        return np.hstack(hidden_outputs)
