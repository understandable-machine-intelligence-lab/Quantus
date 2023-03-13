from pydantic import BaseSettings, Field


class Config(BaseSettings):
    # Use XLA compiler for TF functions, which promises to provide a noticeable speed-up for
    # "a lot of small functions" case compared with regular Grappler. However, not supported on macOS.
    use_xla: bool = Field(default=False, env="USE_XLA")


config = Config()
