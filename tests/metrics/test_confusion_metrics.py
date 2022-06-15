from glob import glob
import torch
import numpy as np
from PIL import Image

import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers import perturb_func
from ...quantus.helpers.explanation_func import explain

from tensorflow.keras.datasets import cifar10
from torchvision.models.resnet import resnet18
from torchvision import transforms

@pytest.fixture
def load_artificial_attribution():
    """Build an artificial attribution map"""
    zeros = np.zeros((1, 28, 28))
    ones = np.ones((1, 28, 28))
    mosaics_list = []
    images = [zeros, ones]
    indices_list = [tuple([0,0,1,1]), tuple([1,1,0,0]), tuple([0,1,0,1]), tuple([1,0,1,0])]
    for indices in indices_list:
        first_row = np.concatenate((images[indices[0]], images[indices[1]]), axis=1)
        second_row = np.concatenate((images[indices[2]], images[indices[3]]), axis=1)
        mosaic = np.concatenate((first_row, second_row), axis=2)
        mosaics_list.append(mosaic)
    return np.array(mosaics_list)


@pytest.fixture()
def load_mnist_adaptive_lenet_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetAdaptivePooling(input_shape=(1, 28, 28))
    model.load_state_dict(
        torch.load("tutorials/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def load_mnist_mosaics():
    """Load a batch of MNIST digits and build mosaics from them"""
    x_batch = torch.as_tensor(
        np.loadtxt("tutorials/assets/mnist_x").reshape(124, 1, 28, 28),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        np.loadtxt("tutorials/assets/mnist_y"), dtype=torch.int64
    ).numpy()
    mosaics_returns = mosaic_creation(images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777)
    all_mosaics, mosaic_indices_list, mosaic_labels_list, p_batch_list, target_list = mosaics_returns
    return {
        "x_batch": all_mosaics,
        "y_batch": target_list,
        "p_batch": p_batch_list,
    }


@pytest.fixture()
def load_cifar10_adaptive_lenet_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNetAdaptivePooling(input_shape=(3, 32, 32))
    model.load_state_dict(
        torch.load("tutorials/assets/cifar10", map_location="cpu", pickle_module=pickle)
    )
    return model


@pytest.fixture
def load_cifar10_mosaics():
    """Load a batch of Cifar10 and build mosaics from them"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_batch = torch.as_tensor(
        x_train[:124, ...].reshape(124, 3, 32, 32),
        dtype=torch.float,
    ).numpy()
    y_batch = torch.as_tensor(
        y_train[:124].reshape(124), dtype=torch.int64
    ).numpy()
    mosaics_returns = mosaic_creation(images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777)
    all_mosaics, mosaic_indices_list, mosaic_labels_list, p_batch_list, target_list = mosaics_returns
    return {
        "x_batch": all_mosaics,
        "y_batch": target_list,
        "p_batch": p_batch_list,
    }


@pytest.fixture()
def load_imagenet_resnet18_model():
    """Load a pre-trained ResNet18 classification model (architecture at quantus/helpers/models)."""
    model = resnet18(pretrained=True)
    return model


@pytest.fixture()
def load_imagenet_mosaics():
    """Load a batch of ImageNet and build mosaics from it."""

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with open("tutorials/assets/imagenet_images/imagenet1000_clsid_to_labels.txt") as mapper_file:
        label_map = {line.split(':')[0]: idx for idx, line in enumerate(mapper_file.readlines())}

    x_validation, y_validation = [], []
    for image_path in glob("tutorials/assets/imagenet_images/images/*/*"):
        input_img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(input_img)
        input_batch = input_tensor.unsqueeze(0)
        x_validation.append(input_batch)

        label_str = os.path.basename(os.path.dirname(image_path))
        label_int = label_map[label_str]
        y_validation.append(label_int)

    x_batch = torch.concat(x_validation).numpy()
    y_batch = np.asarray(y_validation)

    mosaics_returns = mosaic_creation(images=x_batch, labels=y_batch, mosaics_per_class=10, seed=777)
    all_mosaics, mosaic_indices_list, mosaic_labels_list, p_batch_list, target_list = mosaics_returns
    return {
        "x_batch": all_mosaics,
        "y_batch": target_list,
        "p_batch": p_batch_list,
    }


@pytest.mark.confusion
@pytest.mark.parametrize(
    "model,mosaic_data,a_batch,params,expected",
    [
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "Gradient",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "GradientShap",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "GradientShap",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "IntegratedGradients",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "InputXGradient",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "InputXGradient",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "Occlusion",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "FeatureAblation",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "GradCam",
                    "gc_layer": "model._modules.get('conv_2')",
                    "abs": True,
                    "interpolate": (56, 56),
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_cifar10_adaptive_lenet_model"),
                lazy_fixture("load_cifar10_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "GradCam",
                    "gc_layer": "model._modules.get('conv_2')",
                    "pos_only": True,
                    "interpolate": (64, 64),
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_imagenet_resnet18_model"),
                lazy_fixture("load_imagenet_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "GradCam",
                    "gc_layer": "model._modules.get('layer4')[-1]",
                    "pos_only": True,
                    "interpolate": (448, 448),
                    "interpolate_mode": "bilinear",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                lazy_fixture("load_mnist_mosaics"),
                None,
                {
                    "explain_func": explain,
                    "method": "Control Var. Sobel Filter",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                None,
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                {
                    "x_batch": None,
                    "y_batch": None,
                    "p_batch": None,
                },
                None,
                {
                    "explain_func": explain,
                    "method": "Gradient",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                {'exception': ValueError},
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                {
                    "x_batch": np.ones((4, 1, 56, 56)),
                    "y_batch": np.ones(4),
                    "p_batch": [tuple([0,0,1,1]), tuple([1,1,0,0]), tuple([0,1,0,1]), tuple([1,0,1,0])],
                },
                lazy_fixture("load_artificial_attribution"),
                {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                {"value": 1},
        ),
        (
                lazy_fixture("load_mnist_adaptive_lenet_model"),
                {
                    "x_batch": np.ones((4, 1, 56, 56)),
                    "y_batch": np.ones(4),
                    "p_batch": [tuple([1, 1, 0, 0]), tuple([0, 0, 1, 1]), tuple([1, 0, 1, 0]), tuple([0, 1, 0, 1])],
                },
                lazy_fixture("load_artificial_attribution"),
                {
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                {"value": 0},
        ),
    ],
)
def test_focus(
        model: Optional[ModelInterface],
        mosaic_data: Dict[str, Union[np.ndarray, list]],
        a_batch: Optional[np.ndarray],
        params: dict,
        expected: Optional[dict],
):
    x_batch, y_batch, p_batch = (
        mosaic_data["x_batch"],
        mosaic_data["y_batch"],
        mosaic_data["p_batch"]
    )
    metric = Focus(**params)

    if expected and "exception" in expected:
        with pytest.raises(expected["exception"]):
            metric(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                p_batch=p_batch,
                **params,
            )
        return

    p_batch_len = len(p_batch)
    while len(p_batch) > 0:
        if x_batch is not None:
            x_minibatch, x_batch = x_batch[:10], x_batch[10:]
        else:
            x_minibatch = None
        if y_batch is not None:
            y_minibatch, y_batch = y_batch[:10], y_batch[10:]
        else:
            y_minibatch = None
        if a_batch is not None:
            a_minibatch, a_batch = a_batch[:10], a_batch[10:]
        else:
            a_minibatch = None
        p_minibatch, p_batch = p_batch[:10], p_batch[10:]
        metric(
            model=model,
            x_batch=x_minibatch,
            y_batch=y_minibatch,
            a_batch=a_minibatch,
            p_batch=p_minibatch,
            **params,
        )

    scores = metric.last_results
    assert len(scores) == p_batch_len, "Test failed."
    assert all([0 <= score <= 1 for score in scores]), "Test failed."
    if expected and "value" in expected:
        assert all((score == expected["value"]) for score in scores), "Test failed."