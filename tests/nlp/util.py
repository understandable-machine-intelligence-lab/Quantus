import pytest
import platform

skip_on_apple_silicon = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.processor() == "arm",
    reason="Skip test on apple silicon due to missing support for dependencies.",
)
