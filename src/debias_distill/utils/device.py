"""Device selection."""

import torch


def get_device(preferred: str = "mps") -> torch.device:
    """Return the preferred accelerator if available, else CPU."""
    raise NotImplementedError
