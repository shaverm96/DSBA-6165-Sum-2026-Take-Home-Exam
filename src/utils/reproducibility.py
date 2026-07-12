from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to use for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    """Return the most capable device available for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
