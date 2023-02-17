from __future__ import annotations

import torch
import numpy as np
from typing import List, Generator, TypeVar, Callable, Any
from copy import deepcopy

T = TypeVar("T")


def get_hidden_representations(
        model: torch.nn.Module,
        x: T,
        forward_fn: Callable[[torch.nn.Module, T], Any],
        device: str,
        layer_names: List[str],
        layer_indices: List[int],
) -> np.ndarray:
    all_layers = [*model.named_modules()]
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
                l[1], (model.__class__, torch.nn.Sequential)
            ),
            all_layers,
        )
    )

    batch_size = len(x)
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
        forward_fn(model, torch.Tensor(x).to(device))

    # Cleanup.
    [i.remove() for i in new_hooks]
    return np.hstack(hidden_outputs)


def get_random_layer_generator(model: torch.nn.Module, order: str, seed: int) -> Generator:
    original_parameters = model.state_dict()
    random_layer_model = deepcopy(model)

    modules = [
        layer
        for layer in random_layer_model.named_modules()
        if (hasattr(layer[1], "reset_parameters"))
    ]

    if order == "top_down":
        modules = modules[::-1]

    for module in modules:
        if order == "independent":
            random_layer_model.load_state_dict(original_parameters)
        torch.manual_seed(seed=seed + 1)
        module[1].reset_parameters()
        yield module[0], random_layer_model
