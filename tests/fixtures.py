from typing import Tuple
import pytest
import torch
import torchvision
import numpy as np

# CASES
data_cases = {"a_subset_half_s": None,
              "s_subset_half_a": None,
              "a_equal_s": None,
              "a_geq_s": None,
              "a_empty_s": None,
              }

data = {}


#print(metric_edit(_, np.random.randn(10, 3, 24, 24), np.random.randint(0, 10, size=10), np.random.randn(10, 1, 24, 24), np.ones((10, 1, 24, 24))))

@pytest.fixture(autouse=True)
def model():
    return torchvision.models.resnet18(pretrained=True)

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
