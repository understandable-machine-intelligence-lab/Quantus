"""
In this suite we test the cases, when user has installed quantus with only tensorflow (exclusive) or torch,
with or without NLP support. The idea is to catch references on conditional libraries, and verify Quantus is useable
in all extra-installations variants. The test are mostly just copy-pasted from test_evaluation.py.
These are run in separate GitHub actions, as they take a lot of time.
Run with:
- tox run -e tf_only
- tox run -e torch_only
- tox run -e tf_nlp
- tox run -e torch_nlp
"""

from importlib import util
import pytest


def assert_tf_not_installed():
    assert util.find_spec("tensorflow") is None
    assert util.find_spec("keras") is None


def assert_nlp_libraries_not_installed():
    assert util.find_spec("transformers") is None
    assert util.find_spec("nlpaug") is None
    assert util.find_spec("nltk") is None


def assert_torch_not_installed():
    assert util.find_spec("torch") is None
    assert util.find_spec("captum") is None


skip_if_torch_and_tf_installed = pytest.mark.skipif(
    util.find_spec("tensorflow") is not None and util.find_spec("torch") is not None,
    reason="Integration tests must run only with 1 DNN framework installed.",
)


@pytest.mark.integration_test
@skip_if_torch_and_tf_installed
def test_base_tf_installation():
    assert_torch_not_installed()
    assert_nlp_libraries_not_installed()


@pytest.mark.integration_test
@skip_if_torch_and_tf_installed
def test_base_torch_installation():
    assert_tf_not_installed()
    assert_nlp_libraries_not_installed()


@pytest.mark.integration_test
@skip_if_torch_and_tf_installed
def test_tf_nlp_installation():
    assert_torch_not_installed()


@pytest.mark.integration_test
@skip_if_torch_and_tf_installed
def test_torch_nlp_installation():
    assert_tf_not_installed()
