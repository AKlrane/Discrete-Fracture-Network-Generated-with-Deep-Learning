import torch


def mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not mps_available():
        return torch.device("cpu")
    return torch.device(requested)


def lightning_accelerator(requested: str) -> tuple[str, int]:
    device = select_device(requested)
    if device.type == "cuda":
        return "gpu", 1
    if device.type == "mps":
        return "mps", 1
    return "cpu", 1
