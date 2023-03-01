import pytest
import sys


skip_on_python_10 = pytest.mark.skipif(
    sys.version_info[1] == 10,
    reason="Test fail on python 3.10."
)