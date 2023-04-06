import pytest


@pytest.mark.partial_installation
def test_only_tensorflow_installed():
    import quantus


@pytest.mark.partial_installation
def test_only_torch_installed():
    import transformers


@pytest.mark.partial_installation
def test_transformers_with_torch_installed():
    import tensorflow


@pytest.mark.partial_installation
def test_transformers_with_tensorflow_installed():
    import tensorflow