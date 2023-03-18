# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from pydantic import BaseSettings, Field
import logging

"""Library-wide configuration, which can be overriden by ENV variables."""


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

# Set PRNG seed manually for reproducibility.
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
