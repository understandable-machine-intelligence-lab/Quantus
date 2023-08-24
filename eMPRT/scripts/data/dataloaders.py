import os
import random

import torch.utils.data as tdata

DATALOADER_MAPPING = {
    "adience-age": tdata.DataLoader,
    "adience-gender": tdata.DataLoader,
    "adience-age-reduanspreproc": tdata.DataLoader,
    "adience-gender-reduanspreproc": tdata.DataLoader,
    "isic2019": tdata.DataLoader,
    "imagenet": tdata.DataLoader,
    "imagenet-bbox": tdata.DataLoader,
    "pascalvoc": tdata.DataLoader,
    "imagenet-unlearnable": tdata.DataLoader,
    "mnist": tdata.DataLoader,
    "fashionmnist": tdata.DataLoader,
    "cifar10": tdata.DataLoader,
    "cifar10-transfer": tdata.DataLoader,
    "cifar10-unlearnable": tdata.DataLoader,
    "cifar100": tdata.DataLoader,
    "toy": tdata.DataLoader,
    "iris": tdata.DataLoader,
    "gaussquant": tdata.DataLoader,
    "gridblobs": tdata.DataLoader,
}

def get_dataloader(dataset_name, dataset, batch_size, shuffle):
    """
    selects the correct dataloader for the dataset
    """

    # Check if dataset_name is valid
    if dataset_name not in DATALOADER_MAPPING:
        raise ValueError("Dataloader for dataset '{}' not supported.".format(dataset_name))

    # Load correct dataloader
    dataloader = DATALOADER_MAPPING[dataset_name](
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Return dataset
    return dataloader