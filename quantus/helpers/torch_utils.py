import torch


def choose_hardware_acceleration() -> torch.device:
    """Choose device with highest compute capabilities."""
    if torch.cuda.is_available():
        device = "cuda:0"
    elif is_mps_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)


def is_mps_available() -> bool:
    if hasattr(torch.backends, "mps"):
        return torch.backends.mps.is_available()
    return False