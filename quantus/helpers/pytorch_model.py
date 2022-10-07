"""This model creates the ModelInterface for PyTorch."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from contextlib import suppress
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

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
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialisation of PyTorchModel class.

        Parameters
        ----------
        model: Union[torch.nn.Module, tf.keras.Model]
            A model this will be wrapped in the ModelInterface:
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
