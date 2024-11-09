"""This model creates the ModelInterface for PyTorch."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
import copy
import logging
import warnings
from contextlib import suppress
from copy import deepcopy
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    TypedDict,
)


import numpy as np
import numpy.typing as npt
import torch
import sys
from torch import nn

from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface


if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


class PyTorchModel(ModelInterface[nn.Module]):
    """Interface for torch models."""

    def __init__(
        self,
        model: nn.Module,
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
        # self.channel_first = channel_first
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            model_predict_kwargs=model_predict_kwargs,
        )
        self.device = device

    @lru_cache(maxsize=None)
    def _get_last_softmax_layer_index(self) -> Optional[int]:
        """
        Returns the index of the last module of torch.nn.Softmax type in the list of model children.
        If no softmax module is found, returns None.
        """
        modules = list(self.model.modules())
        for i in range(-1, -len(modules), -1):
            if isinstance(modules[i], torch.nn.Softmax):
                return i
        return None

    @lru_cache(maxsize=None)
    def _get_model_with_linear_top(self) -> torch.nn.Module:
        """
        In a case model has a softmax module, the last torch.nn.Softmax module in the self.model.modules() list is
        replaced with torch.nn.Identity().
        Iterates through named modules in reverse order (from the last to the first), for the first module of
        torch.nn.Softmax type, the module's name is then used to replace the module with torch.nn.Identity() in
        the original model's copy using setattr.
        """
        linear_model = copy.deepcopy(self.model)

        for named_module in list(linear_model.named_modules())[::-1]:
            if isinstance(named_module[1], torch.nn.Softmax):
                setattr(linear_model, named_module[0], torch.nn.Identity())

                logging.info(
                    "Argument softmax=False passed, but the passed model contains a module of type "
                    "torch.nn.Softmax. Module {} has been replaced with torch.nn.Identity().",
                    named_module[0],
                )
                break

        return linear_model

    def _obtain_predictions(
        self,
        x: Union[
            torch.Tensor,
            npt.ArrayLike,
            Mapping[str, Union[torch.Tensor, npt.ArrayLike]],
        ],
        model_predict_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if safe_isinstance(self.model, "transformers.modeling_utils.PreTrainedModel"):

            if not is_batch_encoding_like(x):
                raise ValueError(
                    "When using HuggingFace pretrained models, please use Tokenizers output for `x` "
                    "or make sure you're passing a dict with input_ids and attention_mask as keys"
                )

            x = {k: torch.as_tensor(v, device=self.device) for k, v in x.items()}
            pred = self.model(**x, **model_predict_kwargs).logits
            if self.softmax:
                return torch.softmax(pred, dim=-1)
            return pred

        elif isinstance(self.model, nn.Module):
            pred_model = self.get_softmax_arg_model()
            return pred_model(torch.Tensor(x).to(self.device), **model_predict_kwargs)
        else:
            raise ValueError("Predictions cant be null")

    def get_softmax_arg_model(self) -> torch.nn.Module:
        """
        Returns model with last layer adjusted accordingly to softmax argument.
        If the original model has softmax activation as the last layer and softmax=false,
        the layer is removed.
            +----------------------------------------------+----------------+-------------------+
            |                                              | softmax = true |  softmax = false  |
            +----------------------------------------------+----------------+-------------------+
            | torch.nn.Softmax LAST in model.modules()     | Do softmax  (1)| Remove softmax (4)|
            +----------------------------------------------+----------------+-------------------+
            | torch.nn.Softmax NOT LAST in model.modules() | Add softmax (2)| Do nothing     (5)|
            +----------------------------------------------+----------------+-------------------+
            | torch.nn.Softmax NOT in model.modules()      | Add softmax (3)| Do nothing     (6)|
            +----------------------------------------------+----------------+-------------------+

        (cells numbers according to Case N comments in the method)
        """

        # last_softmax is the index of the last module which is of softmax type in the list of model children
        # or None if no softmax layer is found
        last_softmax = self._get_last_softmax_layer_index()

        if self.softmax and last_softmax == -1:
            return self.model  # Case 1

        if self.softmax and not last_softmax:
            logging.info(
                "Argument softmax=True passed, but the passed model contains no module of type "
                "torch.nn.Softmax. torch.nn.Softmax module is added as the output layer."
            )
            return torch.nn.Sequential(self.model, torch.nn.Softmax(dim=-1))  # Case 3

        if not self.softmax and not last_softmax:
            return self.model  # Case 6

        warnings.warn(
            "The combination of the value of the passed softmax argument and the passed model potentially requires "
            "adjusting the model's modules. Make sure that the torch.nn.Softmax layer is the last module in the list "
            "of model's children (self.model.modules()) if and only if it is the actual last module applied before"
            "output."
        )  # Warning for cases 2, 4, 5

        if self.softmax and last_softmax != -1:
            logging.info(
                "Argument softmax=True passed. The passed model contains a module of type "
                "torch.nn.Softmax, but it is not the last in the list of model's children ("
                "self.model.modules()). torch.nn.Softmax module is added as the output layer."
                "Make sure that the torch.nn.Softmax layer is the last module in the list "
                "of model's children (self.model.modules()) if and only if it is the actual last module "
                "applied before output."
            )

            return torch.nn.Sequential(self.model, torch.nn.Softmax(dim=-1))  # Case 2

        if not self.softmax and last_softmax == -1:
            return self._get_model_with_linear_top()  # Case 4

        return self.model  # Case 5

    def predict(
        self,
        x: Union[npt.ArrayLike, Mapping[str, npt.ArrayLike]],
        grad: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray, BatchEncoding
            A given input that the wrapped model predicts on. This can be either a numpy
            or a BatchEncoding (Tokenizers output from huggingface's Tokenizer library)
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
            pred = self._obtain_predictions(x, model_predict_kwargs)
            if pred.requires_grad:
                return pred.detach().cpu().numpy()
            return pred.cpu().numpy()

    def shape_input(
        self,
        x: np.ndarray,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ) -> np.ndarray:
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
            shape = (1, *shape)

        # If shape not the same, reshape the input
        if shape != x.shape:
            x = x.reshape(*shape)

        # Set channel order according to expected input of model.
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        raise ValueError("Channel first order expected for a torch model.")

    def get_model(self) -> torch.nn.Module:
        """
        Get the original torch model.
        """
        return self.model

    def state_dict(self) -> dict:
        """
        Get a dictionary of the model's learnable parameters.
        """
        return self.model.state_dict()

    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator[Tuple[str, nn.Module], None, None]:
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

        modules = [layer for layer in random_layer_model.named_modules() if (hasattr(layer[1], "reset_parameters"))]

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
    ) -> torch.nn.Module:
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

            modules = [layer for layer in new_model.named_modules()]
            module = modules[1]

            delta = torch.zeros(size=shape).fill_(input_shift)
            fw = module[1].forward(delta)[0]

            for i in range(module[1].out_channels):
                if self.channel_first:
                    module[1].bias[i] = torch.nn.Parameter(2 * module[1].bias[i] - torch.unique(fw[i])[0])
                else:
                    module[1].bias[i] = torch.nn.Parameter(2 * module[1].bias[i] - torch.unique(fw[..., i])[0])

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
        As the exact definition of "internal model representation" is left out in the original paper
        (see: https://arxiv.org/pdf/2203.06877.pdf), we make the implementation flexible.
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
        positive_layer_indices = [i if i >= 0 else num_layers + i for i in layer_indices]

        if layer_names is None:
            layer_names = []

        def is_layer_of_interest(layer_index: int, layer_name: str):
            if layer_names == [] and positive_layer_indices == []:
                return True
            return layer_index in positive_layer_indices or layer_name in layer_names

        # skip modules defined by subclassing API.
        hidden_layers = list(  # type: ignore
            filter(
                lambda layer: not isinstance(layer[1], (self.model.__class__, torch.nn.Sequential)),
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

    @property
    def random_layer_generator_length(self) -> int:
        return len([i for i in self.model.named_modules() if (hasattr(i[1], "reset_parameters"))])


def safe_isinstance(obj: Any, class_path_str: Union[Iterable[str], str]) -> bool:
    """Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    -------
    bool: True if isinstance is true and the package exists, False otherwise

    """
    # Taken from https://github.com/shap/shap/blob/dffc346f323ff8cf55f39f71c613ebd00e1c88f8/shap/utils/_general.py#L197

    if isinstance(class_path_str, str):
        class_path_str = [class_path_str]

    # try each module path in order
    for class_path_str in class_path_str:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


class BatchEncodingLike(TypedDict):
    input_ids: Union[torch.Tensor, npt.ArrayLike]
    attention_mask: Union[torch.Tensor, npt.ArrayLike]


def is_batch_encoding_like(x: Any) -> TypeGuard[BatchEncodingLike]:
    # BatchEncoding is the default output from Tokenizers which contains
    # necessary keys such as `input_ids` and `attention_mask`.
    # It is also possible to pass a Dict with those keys.
    if safe_isinstance(x, "transformers.tokenization_utils_base.BatchEncoding"):
        return True

    elif isinstance(x, Mapping) and "input_ids" in x and "attention_mask" in x:
        return True

    else:
        return False
