from pydantic import BaseSettings, Field
import logging


class Config(BaseSettings):
    # Use XLA compiler for TF functions, which promises to provide a noticeable speed-up for
    # "a lot of small functions" case compared with regular Grappler. However, not supported on macOS.
    use_xla: bool = Field(default=False, env="USE_XLA")
    log_level: str = Field(default="WARN", env="Q_LOG_LEVEL")
    prng_seed: int = Field(default=42, env="Q_PRNG_SEED")


config = Config()

logging.basicConfig(
    format="%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    level=logging.getLevelName(config.log_level),
)

import numpy as np

np.random.seed(config.prng_seed)

try:
    import tensorflow as tf

    tf.random.set_seed(config.prng_seed)
except ModuleNotFoundError:
    pass

try:
    import torch

    torch.manual_seed(config.prng_seed)
except ModuleNotFoundError:
    pass
