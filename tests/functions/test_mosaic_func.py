import pytest
from pytest_lazyfixture import lazy_fixture

from quantus.functions.mosaic_func import *


@pytest.mark.mosaic_func
@pytest.mark.parametrize(
    "data,params",
    [
        (
            lazy_fixture("load_mnist_images"),
            {"mosaics_per_class": 4, "seed": 777},
        ),
        (
            lazy_fixture("load_cifar10_images"),
            {"mosaics_per_class": 4, "seed": 777},
        ),
        (
            lazy_fixture("load_mnist_images"),
            {"mosaics_per_class": 10, "seed": 777},
        ),
    ],
)
def test_mosaic_func(
    data: np.ndarray,
    params: dict,
):
    x_batch, y_batch = (data["x_batch"], data["y_batch"])
    return_params = mosaic_creation(
        images=x_batch,
        labels=y_batch,
        mosaics_per_class=params["mosaics_per_class"],
        seed=params["seed"],
    )
    (
        all_mosaics,
        mosaic_indices_list,
        mosaic_labels_list,
        p_batch_list,
        target_list,
    ) = return_params

    _, _, width, height = x_batch.shape
    for mosaic in all_mosaics:
        _, width_mosaic, height_mosaic = mosaic.shape
        assert width_mosaic / width == 2, "Test failed."
        assert height_mosaic / height == 2, "Test failed."

    for mosaic, mosaic_indices in zip(all_mosaics, mosaic_indices_list):
        assert len(mosaic_indices) == 4
        assert np.all(
            np.equal(mosaic[:, :width, :height], x_batch[mosaic_indices[0]])
        ), "Test failed."
        assert np.all(
            np.equal(mosaic[:, width:, :height], x_batch[mosaic_indices[1]])
        ), "Test failed."
        assert np.all(
            np.equal(mosaic[:, :width, height:], x_batch[mosaic_indices[2]])
        ), "Test failed."
        assert np.all(
            np.equal(mosaic[:, width:, height:], x_batch[mosaic_indices[3]])
        ), "Test failed."

    for mosaic_labels in mosaic_labels_list:
        assert len(mosaic_labels) == 4, "Test failed."
        assert (
            len(set([x for x in mosaic_labels if mosaic_labels.count(x) == 2])) > 0
        ), "Test failed."

    for p_batch in p_batch_list:
        assert len(p_batch) == 4, "Test failed."
        assert (
            len(set([x for x in p_batch if p_batch.count(x) == 2])) == 2
        ), "Test failed."

    for mosaic_labels, p_batch, target in zip(
        mosaic_labels_list, p_batch_list, target_list
    ):
        mosaic_labels = np.array(mosaic_labels)
        p_batch = np.array(p_batch)
        assert np.unique(mosaic_labels[p_batch == 1]) == target, "Test failed."

    for target in set(target_list):
        assert target_list.count(target) == params["mosaics_per_class"], "Test failed."
