import torch
import torchvision.transforms as T

TRANSFORM_MAP = {
    "adience-age": {
            "train": [T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize(256),
                     T.FiveCrop(224),
                     T.Lambda(lambda crops: [crop for crop in crops]+[T.RandomHorizontalFlip(p=1)(crop) for crop in crops]),
                     T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                     T.Lambda(lambda crops: torch.stack([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), )(crop) for crop in crops]))
                     ],
            "val": [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    },
    "adience-gender": {
            "train": [T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize(256),
                     T.FiveCrop(224),
                     T.Lambda(lambda crops: [crop for crop in crops]+[T.RandomHorizontalFlip(p=1)(crop) for crop in crops]),
                     T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
                     T.Lambda(lambda crops: torch.stack([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), )(crop) for crop in crops]))
                     ],
            "val": [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    },
    "adience-age-reduanspreproc": {
        "train": [
            T.RandomAffine(10, (0.1, 0.1), (0.8, 1.2), 5),
            T.Resize(224),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
        "test": [
            T.Resize(256),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
        "val": [
            T.Resize(256),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
    },
    "adience-gender-reduanspreproc": {
         "train": [
            T.RandomAffine(10, (0.1, 0.1), (0.8, 1.2), 5),
            T.Resize(224),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
        "test": [
            T.Resize(256),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
        "val": [
            T.Resize(256),
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ],
    },
    "pascalvoc": {
        "train": [T.Resize((300, 300)),
                  T.RandomChoice([T.ColorJitter(brightness=(0.80, 1.20)), T.RandomGrayscale(p=0.25)]),
                  T.RandomHorizontalFlip(p=0.25),
                  T.RandomRotation(25),
                  T.ToTensor(),
                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                  ],
        "test": [T.Resize(330),
                 T.CenterCrop(300),
                 T.ToTensor(),
                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ],
        "val": [T.Resize(330),
                 T.CenterCrop(300),
                 T.ToTensor(),
                 T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ],
    },

    "isic2019": {
        # TODO: Reevaluate Transforms
            "train": [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize((224, 224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
    "imagenet": {
            "train": [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize((224, 224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "val": [T.Resize((224, 224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
    "imagenet-bbox": {
        "train": [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
        "test": [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
        "val": [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
    "imagenet-unlearnable": {
        "train": [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
        "test": [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
    "mnist": {
            "train": [T.ToTensor(), T.Normalize((0.5,), (0.5,))],
            "test": [T.ToTensor(), T.Normalize((0.5,), (0.5,))],
    },
    "fashionmnist": {
            "train": [T.ToTensor(), T.Normalize((0.5,), (0.5,))],
            "test": [T.ToTensor(), T.Normalize((0.5,), (0.5,))],
    },
    "cifar10": {
            "train": [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
            "test": [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    },
    "cifar10-transfer": {
            "train": [T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
            "test": [T.Resize((224)),  T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    },
    "cifar10-unlearnable": {
        "train": [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
        "test": [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    },
    "cifar100": {
            "train": [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
            "test": [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    },
    "toy": {
            "train": [torch.from_numpy],
            "test": [torch.from_numpy],
    },
    "iris": {
            "train": [torch.from_numpy],
            "test": [torch.from_numpy],
    },
    "gaussquant": {
            "train": [torch.from_numpy],
            "test": [torch.from_numpy],
    },
    "gridblobs": {
            "train": [torch.from_numpy],
            "test": [torch.from_numpy],
    },
}

def get_transforms(dataset_name, mode):
    """
    Gets the correct transforms for the dataset
    """

    # Check if dataset_name is supported
    if dataset_name not in TRANSFORM_MAP:
        raise ValueError("Dataset '{}' not supported.".format(dataset_name))

    # Combine transforms
    transforms = T.Compose(TRANSFORM_MAP[dataset_name][mode])

    # Return transforms
    return transforms
