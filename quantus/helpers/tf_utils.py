import tensorflow as tf
from tensorflow import keras
import platform


def is_xla_compatible_platform() -> bool:
    """Determine if model is xla-compatible."""
    return not (
            platform.system() == "Darwin" and "arm" in platform.processor().lower()
    )


def is_xla_compatible_model(model: keras.Model) -> bool:
    """Determine if model and are platform xla-compatible."""
    return is_xla_compatible_platform() and not isinstance(
        model.distribute_strategy,
        (
            tf.compat.v1.distribute.experimental.TPUStrategy,
            tf.distribute.TPUStrategy,
        ),
    )
