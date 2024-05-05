import pytest
import numpy as np
from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def mock_prediction_changed(mocker: MockerFixture):
    # Event tough it relies on numpy PRNG, because of setting seed manually, the sequence of
    # generated pseudo-logits will always stay the same.
    rng = np.random.default_rng(42)

    def mock_predict(self, x_batch, *args):
        # Event though technically, this is a stub, we called fixture and function `mock_`
        # to be aligned with pytest_mock naming.
        batch_size = len(x_batch)
        y_batch = rng.uniform(size=(batch_size, 10), low=-100, high=100)
        return y_batch

    mocker.patch("quantus.helpers.model.tf_model.TensorFlowModel.predict", mock_predict)
    mocker.patch(
        "quantus.helpers.model.pytorch_model.PyTorchModel.predict", mock_predict
    )
    yield
    # Restore original behaviour after test finished execution.
    mocker.resetall(side_effect=True, return_value=True)
