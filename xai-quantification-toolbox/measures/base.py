"""This module implements the base class for creating evaluation measures."""
from typing import Optional, Any, Dict, Union, List
import tensorflow as tf
import torch
import numpy as np


class Measure:
    """
    This class is the base class for creating evaluation measures.
    The measures outputs at minimum one numerical value per sample and explanation method.
    If more than one sample and/or more than one explanation method is evaluated, then the dimensions increases.
    """

    def __init__(self,
                 name: Optional[str] = "Measure",
                 **params: Dict[str, Any]):
        assert isinstance(name, str)
        assert isinstance(params, Dict)
        self.name = name
        self.params = params

    def fit(self,
            model: Union[tf.keras.models.Model, torch.nn.Module, None],
            inputs: Union[tf.Tensor, torch.Tensor, np.array],
            targets: Union[tf.Tensor, torch.Tensor, np.array, None],
            attributions: Union[tf.Tensor, torch.Tensor, np.array, None]) -> List[float, np.array]:
        raise NotImplementedError('Implement a compute method.')

    @property
    def get_params(self) -> Dict[str, Union[Optional[str], Optional[int], Optional[float], List]]:
        for k, v in self.params.items():
            print(k, v)
        return self.params

    def set_params(self, key: Optional[str], value: Any) -> Dict[str, Any]:
        self.params[key] = value
        return self.params


class ModularMeasure:

    def __init__(self,
                 name: Optional[str] = "Measure",
                 **params: Dict[str, Any]):
        """ Initialize Measure. """
        assert isinstance(name, str)
        assert isinstance(params, Dict)
        self.name = name
        self.params = params

    def __call__(self,
                 model,
                 inputs: np.array,
                 targets: Union[np.array, int, None],
                 attributions: Union[np.array, None],
                 ):
        """ Makes computation for given data. Return float/Array per Sample. """

        raise NotImplementedError("Implementation of the Measure missing.")

    @property
    def get_params(self) -> Dict[str, Union[Optional[str], Optional[int], Optional[float], List]]:
        for k, v in self.params.items():
            print(k, v)
        return self.params

    def set_params(self, key: Optional[str], value: Any) -> Dict[str, Any]:
        self.params[key] = value
        return self.params
