from typing import Tuple
import pytest
import torch
import torchvision
import numpy as np
from ..quantus.helpers.models import LeNet

'''
@pytest.fixture
def load_mnist_data(scope="session"):
    pass


#print(metric_edit(_, np.random.randn(10, 3, 24, 24), np.random.randint(0, 10, size=10), np.random.randn(10, 1, 24, 24), np.ones((10, 1, 24, 24))))

@pytest.fixture(autouse=True, scope="session")
def load_mnist_modeLeNet():
    model = LeNet()
    model.load_state_dict(torch.load("tutorials/assets/mnist"))
    return model

'''

#### METRICS

@pytest.fixture
def all_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def all_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def all_in_gt_non_normalised():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 20, size=(10, 1, 224, 224))
    s_batch[:, :, 50:150, 50:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 25
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def all_in_gt_seg_bigger():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:150, 0:150] = 1.0
    a_batch[:, :, 50:150, 50:150] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def none_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def none_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 100:200, 100:200] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def none_in_gt_fourth():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:112, 0:112] = 1.0
    a_batch[:, :, 112:224, 112:224] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}


@pytest.fixture
def half_in_gt_zeros():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def half_in_gt():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.random.uniform(0, 0.1, size=(10, 1, 224, 224))
    s_batch[:, :, 50:100, 50:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}

@pytest.fixture
def half_in_gt_zeros_bigger():
    s_batch = np.zeros((10, 1, 224, 224))
    a_batch = np.zeros((10, 1, 224, 224))
    s_batch[:, :, 0:100, 0:100] = 1.0
    a_batch[:, :, 0:100, 75:100] = 1.0
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch,
            "s_batch": s_batch}


@pytest.fixture
def almost_uniform():
    a_batch = np.random.uniform(0, 0.01, size=(10, 1, 224, 224))
    return {"x_batch": np.random.randn(10, 3, 224, 224),
            "y_batch": np.random.randint(0, 10, size=10),
            "a_batch": a_batch}


#### SIMILARITY TESTS

@pytest.fixture
def atts_half():
    return {"a": np.array([-1, 1, 1]),
            "b": np.array([0, 0, 2])}

@pytest.fixture
def atts_diff():
    return {"a": np.array([0, 1, 0, 1]),
            "b": np.array([1, 2, 1, 0])}


@pytest.fixture
def atts_same():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a,
            "b": a}

@pytest.fixture
def atts_same_linear():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a,
            "b": a*3}

@pytest.fixture()
def atts_inverse():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a,
            "b": a*-3}


@pytest.fixture
def atts_lip_same():
    return {"a": np.array([-1, 1, 1]),
            "b": np.array([0, 0, 2]),
            "c": np.array([-1, 1, 1]),
            "d": np.array([0, 0, 2])}

@pytest.fixture
def atts_lip_diff():
    return {"a": np.array([-1, 1, 1]),
            "b": np.array([0, 0, 2]),
            "c": np.array([-1, 1, 1]),
            "d": np.array([0, 0, 2])}


@pytest.fixture
def atts_ssim_same():
    a = np.random.uniform(0, 0.1, size=(10))
    return {"a": a,
            "b": a}


@pytest.fixture
def atts_ssim_diff():
    return {"a": np.zeros((16, 16)),
            "b": np.ones((16, 16))}

#### NORMALISE_FUNC

@pytest.fixture
def atts_normalise():
    return np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, -1.0])


#### NORM_FUNC

@pytest.fixture
def atts_norm_ones():
    return np.ones((10))


@pytest.fixture
def atts_norm_fill():
    return np.array([1, 2, 3, 4, 10])


#### PERTURB_FUNC

@pytest.fixture
def input_pert_1d():
    return np.random.uniform(0, 0.1, size=(1, 3, 224, 224)).flatten()

@pytest.fixture
def input_pert_3d():
    return np.random.uniform(0, 0.1, size=(1, 3, 224, 224))



