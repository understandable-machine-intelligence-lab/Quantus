import pytest
import platform
from os import environ

skip_on_apple_silicon = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.processor() == "arm",
    reason="Skip test on apple silicon due to missing support for dependencies.",
)


skip_in_ci = pytest.mark.skipif(
    "CI" in environ,
    reason="Skip test because they will fail in CI due to OOM.",
)
