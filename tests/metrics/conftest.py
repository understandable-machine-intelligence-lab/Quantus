import pytest
import numpy as np
from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def mock_prediction_changed(mocker: MockerFixture):
    def mock_predict(self, x_batch, *args):
        batch_size = len(x_batch)
        y_batch = np.random.uniform(size=(batch_size, 1000))
        return y_batch

    mocker.patch("quantus.helpers.model.tf_model.TensorFlowModel.predict", mock_predict)
    mocker.patch(
        "quantus.helpers.model.pytorch_model.PyTorchModel.predict", mock_predict
    )
    yield
    mocker.resetall()