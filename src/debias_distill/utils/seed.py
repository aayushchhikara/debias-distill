"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python `random`, numpy and torch (CPU and MPS).

    MPS kernels are not guaranteed to be bit-for-bit deterministic, so results
    should be compared across several seeds rather than assumed identical.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
