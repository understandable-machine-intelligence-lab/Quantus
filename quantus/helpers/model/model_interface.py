"""This model implements the basics for the ModelInterface class."""
# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import warnings
from importlib import util
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Union, Generator, TypeVar, Generic

import numpy as np

if util.find_spec("tensorflow"):
    import tensorflow as tf
if util.find_spec("torch"):
    import torch

M = TypeVar("M")


class ModelInterface(ABC, Generic[M]):
    """Base ModelInterface for torch and tensorflow models."""

    def __init__(
        self,
        model: M,
        channel_first: Optional[bool] = True,
        softmax: bool = False,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialisation of ModelInterface class.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A model this will be wrapped in the ModelInterface:
        channel_first: boolean, optional
             Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        """
        self.model = model
        self.channel_first = channel_first
        self.softmax = softmax

        if model_predict_kwargs is None:
            self.model_predict_kwargs = {}
        else:
            self.model_predict_kwargs = model_predict_kwargs

    @abstractmethod
    def get_softmax_arg_model(self):
        """
        Returns model with last layer adjusted accordingly to softmax argument.
        If the original model has softmax activation as the last layer and softmax=false,
        the layer is removed.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.array, **kwargs):
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray
         A given input that the wrapped model predicts on.
        kwargs: optional
            Keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def shape_input(
        self,
        x: np.array,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ):
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
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """
        Get the original torch/tf model.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """
        Get a dictionary of the model's learnable parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator[Tuple[str, M], None, None]:
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.
        """
        raise NotImplementedError

    @abstractmethod
    def add_mean_shift_to_first_layer(
        self,
        input_shift: Union[int, float],
        shape: tuple,
    ):
        """
        Consider the first layer neuron before non-linearity: z = w^T * x1 + b1. We update
        the bias b1 to b2:= b1 - w^T * m. The operation is necessary for Input Invariance metric.


        Parameters
        ----------
        input_shift: Union[int, float]
            Shift to be applied.
        shape: tuple
            Model input shape.

        Returns
        -------
        new_model: torch.nn
            The resulting model with a shifted first layer.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError()

    @property
    @abstractmethod
    def random_layer_generator_length(self) -> int:
        """
        Count number of randomisable layers for `Model Parameter Randomisation`.
        This property is needed to avoid `len(model.get_random_layer_generator())`,
        because meterializing bigger models `num_layers` times in memory at ones
        has shown to cause OOM errors.

        Returns
        -------
        n:
            Number of layers in model, which can be randomised.
        """
        raise NotImplementedError

    @property
    def get_ml_framework_name(self) -> str:
        """
        Identify the framework of the underlying model (PyTorch or TensorFlow).

        Returns
        -------
        str
            A string indicating the framework ('PyTorch', 'TensorFlow', or 'Unknown').
        """
        if util.find_spec("torch"):
            if isinstance(self.model, torch.nn.Module):
                return "torch"
        if util.find_spec("tensorflow"):
            if isinstance(self.model, tf.keras.Model):
                return "tensorflow"
        else:
            warnings.warn("Cannot identify ML framework of the given model.")
            return "unknown"
        return ""
