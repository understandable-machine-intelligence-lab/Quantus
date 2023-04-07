import platform
from functools import lru_cache


@lru_cache
def is_tf_available() -> bool:
    try:
        import tensorflow as tf
        return True
    except ModuleNotFoundError:
        return False


def is_xla_compatible_platform() -> bool:
    """Determine if host is xla-compatible."""
    return not (
            platform.system() == "Darwin" and "arm" in platform.processor().lower()
    )


if is_tf_available():
    import tensorflow as tf

    @tf.function(reduce_retracing=True, jit_compile=is_xla_compatible_platform())
    def ndim(x):
        return tf.size(tf.shape(x))
