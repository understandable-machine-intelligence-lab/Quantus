from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np


def list_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    def should_not_skip(name: str, module: nn.Module):
        # skip modules defined by subclassing and sequential API.
        return not isinstance(module, (model.__class__, nn.Sequential))

    # n is name, m is module.
    return list(filter(lambda i: should_not_skip(*i), model.named_modules()))


def get_random_layer_generator(
    model: nn.Module, order: str = "top_down", seed: int = 42
):
    from quantus.helpers.utils import map_dict

    original_parameters = model.state_dict().copy()

    modules = list_layers(model)

    def randomize_params(w):
        return torch.tensor(
            np.random.permutation(w.detach().cpu().numpy()), device=w.device
        )

    np.random.seed(seed)

    if order == "top_down":
        modules = modules[::-1]

    for module in modules:
        if order == "independent":
            model.load_state_dict(original_parameters)
        params = module[1].state_dict()
        params = map_dict(params, randomize_params)
        module[1].load_state_dict(params)
        yield module[0], model

    # Restore original weights.
    model.load_state_dict(original_parameters)


def random_layer_generator_length(model: nn.Module) -> int:
    modules = [
        module for module in model.modules() if (hasattr(module, "reset_parameters"))
    ]
    return len(modules)
