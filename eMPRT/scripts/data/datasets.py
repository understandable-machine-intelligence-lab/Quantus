import os

import torchvision.datasets as tvisiondata
import torchvision.transforms as T

from . import custom_datasets

DATASET_MAPPING = {
    "adience-age": custom_datasets.CustomDataset,
    "adience-gender": custom_datasets.CustomDataset,
    "isic2019": custom_datasets.CustomDataset,
    "imagenet": custom_datasets.CustomDataset,
    "mnist": tvisiondata.MNIST,
    "cifar10": tvisiondata.CIFAR10,
    "cifar10-transfer": tvisiondata.CIFAR10,
    "cifar100": tvisiondata.CIFAR100,
}

def get_dataset(dataset_name, root_path, transform, mode, **kwargs):
    """
    gets the specified dataset and saves it
    """

    # Check if mode is valid
    if mode not in ["train", "test"]:
        raise ValueError("Mode '{}' not supported. Mode needs to be one of 'train', 'test'".format(mode))

    # Map mode (kinda illegal but so that imagenet works)
    if (dataset_name == "imagenet" or dataset_name == "imagenet-bbox" or dataset_name == "pascalvoc") and mode == "test":
        mode = "val"

    if dataset_name == "imagenet-bbox":
        kwargs["segmentation_transform"] = T.Compose([T.ToTensor(), T.Resize((224, 224))])

    # Check if dataset_name is valid
    if dataset_name not in DATASET_MAPPING:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Adapt root_path
    if DATASET_MAPPING[dataset_name] not in [custom_datasets.CustomDataset]:
        root = os.path.join(root_path, dataset_name)
    else:
        root = root_path

    # Load correct dataset
    if dataset_name not in ["mnist", "fashionmnist", "cifar10", "cifar10-transfer", "cifar100"]:
        dataset = DATASET_MAPPING[dataset_name](
            root = root,
            transform = transform,
            **{
                **kwargs,
                **{
                    "download": True,
                    "train": mode == "train",
                    "mode": mode,
                },
            }
        )
    else:
        dataset = DATASET_MAPPING[dataset_name](
            root=root,
            transform=transform,
            download = True,
            train = mode == "train",
            **kwargs
        )

    # Return dataset
    return dataset