import pytest
from quantus.helpers.tf_utils import is_tf_available
from quantus.helpers.torch_utils import is_torch_available

skip_if_tf_and_torch_available = pytest.mark.skipif(
    is_tf_available() and is_torch_available()
)


@pytest.mark.partial_installation
@skip_if_tf_and_torch_available
def test_only_tensorflow_installed():
    import quantus


@pytest.mark.partial_installation
@skip_if_tf_and_torch_available
def test_only_torch_installed():
    import quantus


@pytest.mark.partial_installation
@skip_if_tf_and_torch_available
def test_transformers_with_torch_installed():
    import quantus


@pytest.mark.partial_installation
@skip_if_tf_and_torch_available
def test_transformers_with_tensorflow_installed():
    import quantus
