import torch


def choose_hardware_acceleration() -> torch.device:
    """Choose device with highest compute capabilities."""
    return torch.device(_choose_hardware_acceleration())


def _choose_hardware_acceleration() -> str:
    if torch.cuda.is_available():
        return "cuda:0"

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"
