import platform
from typing import List, Generator, Protocol, TypeVar
from importlib import util
import numpy as np


def is_tf_available() -> bool:
    return util.find_spec("tensorflow") is not None


def is_xla_compatible_platform() -> bool:
    """Determine if host is xla-compatible."""
    return not (platform.system() == "Darwin" and "arm" in platform.processor().lower())


if is_tf_available():
    import tensorflow as tf
    from tensorflow import keras

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def ndim(x):
        return tf.size(tf.shape(x))

    def list_parameterizable_layers(
        model: keras.Model, flatten_layers: bool = False
    ) -> List[keras.layers.Layer]:
        if flatten_layers:
            layers = list(model._flatten_layers(include_self=False, recursive=True))
        else:
            layers = model.layers
        return list(filter(lambda i: len(i.get_weights()) > 0, layers))

    class TFWrapper(Protocol):
        model: keras.Model

        def state_dict(self) -> List[np.ndarray]:
            ...

        def load_state_dict(self, params: List[np.ndarray]) -> None:
            ...

    T = TypeVar("T", bound=TFWrapper, covariant=True)

    def random_layer_generator(
        model_wrapper: T, order: str = "top_down", seed: int = 42, flatten_layers=False
    ) -> Generator[T, None, None]:
        original_parameters = model_wrapper.state_dict().copy()
        layers = list_parameterizable_layers(model_wrapper.model, flatten_layers)

        np.random.seed(seed)

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                model_wrapper.load_state_dict(original_parameters)

            weights = layer.get_weights()
            layer.set_weights(tf.nest.map_structure(np.random.permutation, weights))
            yield layer.name, model_wrapper
        # Restore original weights.
        model_wrapper.load_state_dict(original_parameters)
