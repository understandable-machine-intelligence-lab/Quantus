import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers import perturb_func
from ...quantus.helpers.explanation_func import explain

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

    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        p_batch=p_batch,
        **params,
    )

    assert len(scores) == len(p_batch)
    assert all([0 <= score <= 1 for score in scores])
    if expected and "value" in expected:
        assert all((score == expected["value"]) for score in scores)
