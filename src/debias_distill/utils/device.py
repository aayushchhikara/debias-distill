"""Device selection."""

import torch


def get_device(preferred: str = "mps") -> torch.device:
    """Return the preferred accelerator if available, else CPU."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
