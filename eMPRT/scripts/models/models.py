import torchvision

TV_MODEL_MAP = {
    "resnet50": torchvision.models.resnet50,
    "resnet34": torchvision.models.resnet34,
    "resnet18": torchvision.models.resnet18,
    "densenet121": torchvision.models.densenet121,
    "vgg16": torchvision.models.vgg16,
}

def get_model(model_name, device):
    """
    Gets the correct model
    """

    # Check if model_name is supported
    if model_name not in TV_MODEL_MAP:
        raise ValueError("Model '{}' is not supported.".format(model_name))

    # Build model
    if model_name in TV_MODEL_MAP:
        model = TV_MODEL_MAP[model_name](pretrained=True)

    # Return model on correct device
    return model.to(device)