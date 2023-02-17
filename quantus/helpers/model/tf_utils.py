from __future__ import annotations

import tensorflow as tf
from keras.models import clone_model
import numpy as np
from typing import Generator, List, Tuple, Callable, TypeVar
from warnings import warn

T = TypeVar("T")

LayersFn = Callable[[tf.keras.Model], List[tf.keras.layers.Layer]]


def build_hidden_representation_model(
        model: tf.keras.Model,
        layer_names: Tuple,
        layer_indices: Tuple,
        layers_fn: LayersFn
) -> tf.keras.Model:
    if layer_names == () and layer_indices == ():
        warn(
            "quantus.TensorFlowModel.get_hidden_layers_representations(...) received `layer_names`=None and "
            "`layer_indices`=None. This will force creation of tensorflow.keras.Model with outputs of each layer"
            " from original model. This can be very computationally expensive."
        )

    def is_layer_of_interest(index: int, name: str) -> bool:
        if layer_names == () and layer_indices == ():
            return True
        return index in layer_indices or name in layer_names

    layers = layers_fn(model)
    outputs_of_interest = []
    for i, layer in enumerate(layers):
        if is_layer_of_interest(i, layer.name):
            outputs_of_interest.append(layer.output)

    if len(outputs_of_interest) == 0:
        raise ValueError("No hidden representations were selected.")

    hidden_representation_model = tf.keras.Model(model.input, outputs_of_interest)
    return hidden_representation_model


def default_list_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    return model.layers


def get_hidden_representations(
        model: tf.keras.Model,
        x: T,
        predict_fn: Callable[[tf.keras.Model, T], List[np.ndarray] | np.ndarray],
        layer_names: List[str],
        layer_indices: List[int],
        layers_fn: LayersFn = None
) -> np.ndarray:

    if layers_fn is None:
        layers_fn = default_list_layers
    num_layers = len(model.layers)

    if layer_indices is None:
        layer_indices = []

    # E.g., user can provide index -1, in order to get only representations of the last layer.
    # E.g., for 7 layers in total, this would correspond to positive index 6.
    positive_layer_indices = [
        i if i >= 0 else num_layers + i for i in layer_indices
    ]
    if layer_names is None:
        layer_names = []

    # List is not hashable, so we pass names + indices as tuples.
    hidden_representation_model = build_hidden_representation_model(
        model,
        tuple(layer_names),
        tuple(positive_layer_indices),
        layers_fn
    )
    internal_representation = predict_fn(hidden_representation_model, x)
    input_batch_size = len(x)

    # If we requested outputs only of 1 layer, keras will already return np.ndarray.
    # Otherwise, keras returns a List of np.ndarray's.
    if isinstance(internal_representation, np.ndarray):
        return internal_representation.reshape((input_batch_size, -1))

    internal_representation = [
        i.reshape((input_batch_size, -1)) for i in internal_representation
    ]
    return np.hstack(internal_representation)


def list_layers_non_zero_weights(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    return [
        _layer
        for _layer in model.layers
        if len(_layer.get_weights()) > 0
    ]


def get_random_layer_generator(
        model: tf.keras.Model,
        order: str = "top_down",
        seed: int = 42,
        layers_fn: LayersFn = None
) -> Generator:
    if layers_fn is None:
        layers_fn = list_layers_non_zero_weights

    original_weights = model.get_weights()
    random_layer_model = clone_model(model)

    layers = layers_fn(random_layer_model)

    if order == "top_down":
        layers = layers[::-1]

    for layer in layers:
        if order == "independent":
            random_layer_model.set_weights(original_weights)
        weights = layer.get_weights()
        np.random.seed(seed=seed + 1)
        layer.set_weights([np.random.permutation(w) for w in weights])
        yield layer.name, random_layer_model
