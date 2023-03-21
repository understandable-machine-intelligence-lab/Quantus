from typing import List, Generator, Tuple
import tensorflow as tf
import numpy as np


def list_parameterizable_layers(
    model: tf.keras.Model, flatten_layers: bool
) -> List[tf.keras.layers.Layer]:
    if flatten_layers:
        layers = list(model._flatten_layers(include_self=False, recursive=True))  # noqa
    else:
        layers = model.layers
    return list(filter(lambda i: len(i.get_weights()) > 0, layers))


def get_random_layer_generator(
    model: tf.keras.Model,
    order: str = "top_down",
    seed: int = 42,
    flatten_layers: bool = False,
) -> Generator[Tuple[str, tf.keras.Model], None, None]:
    original_parameters = model.get_weights().copy()
    layers = list_parameterizable_layers(model, flatten_layers)

    np.random.seed(seed)

    if order == "top_down":
        layers = layers[::-1]

    for layer in layers:
        if order == "independent":
            model.set_weights(original_parameters)

        weights = layer.get_weights()
        layer.set_weights([np.random.permutation(w) for w in weights])
        yield layer.name, model

    # Restore original weights.
    model.set_weights(original_parameters)


def random_layer_generator_length(
    model: tf.keras.Model, flatten_layers: bool = False
) -> int:
    return len(list_parameterizable_layers(model, flatten_layers))
