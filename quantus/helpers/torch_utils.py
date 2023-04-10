from importlib import util
from typing import TypeVar, Tuple, List, Protocol, Dict, Generator

import numpy as np

from quantus.helpers.types import LayerOrderT


def is_torch_available() -> bool:
    return util.find_spec("torch") is not None


if is_torch_available():
    import torch
    import torch.nn as nn

    def choose_hardware_acceleration() -> torch.device:
        """Choose device with highest compute capabilities."""
        if torch.cuda.is_available():
            device = "cuda:0"
        elif is_mps_available():
            device = "mps"
        else:
            device = "cpu"
        return torch.device(device)

    def is_mps_available() -> bool:
        if hasattr(torch.backends, "mps"):
            return torch.backends.mps.is_available()
        return False

    def list_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
        def should_not_skip(name: str, module: nn.Module):
            # skip modules defined by subclassing and sequential API.
            return not isinstance(module, (model.__class__, nn.Sequential))

        # n is name, m is module.
        return list(filter(lambda i: should_not_skip(*i), model.named_modules()))

    class TorchWrapper(Protocol):
        model: nn.Module
        device: torch.device

        def state_dict(self) -> Dict[str, torch.Tensor]:
            ...

        def load_state_dict(self, weights: Dict[str, torch.Tensor]) -> None:
            ...

    T = TypeVar("T", bound=TorchWrapper, covariant=True)

    def random_layer_generator(
        model_wrapper: T,
        order: LayerOrderT = "top_down",
        seed: int = 42,
    ) -> Generator[T, None, None]:
        from quantus.helpers.collection_utils import map_dict

        original_parameters = model_wrapper.state_dict().copy()

        modules = list_layers(model_wrapper.model)

        def randomize_params(w):
            return torch.tensor(
                np.random.permutation(w.detach().cpu().numpy()),
                device=model_wrapper.device,
            )

        np.random.seed(seed)

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                model_wrapper.load_state_dict(original_parameters)
            params = module[1].state_dict()
            params = map_dict(params, randomize_params)
            module[1].load_state_dict(params)
            yield module[0], model_wrapper

        # Restore original weights.
        model_wrapper.load_state_dict(original_parameters)

    def is_torch_model(model):
        if isinstance(model, nn.Module):
            return True

        for attr in ("model", "_model"):
            if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
                return True
        return False

else:

    def is_torch_model(model) -> bool:
        # Torch is not installed, so it is definitely not a torch model.
        return False
