# -*- coding: utf-8 -*-
"""
Cosine beta schedule for variance preserving diffusion models (OOP version).

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

from .base_schedule import NoiseSchedule
from torch import Tensor
import torch
import numpy as np

class CosineSchedule(NoiseSchedule):
    """
    Cosine beta schedule class for variance preserving diffusion models.

    Args:
        s: Small offset parameter to avoid singularities at t=0 (default: 0.008).

    Methods:
        alphas_cumprod(t): Computes cumulative alpha from t using cosine schedule.
        beta(t): Computes beta at time t using cosine schedule.

    Example:
        >>> schedule = CosineSchedule(s=0.008)
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> beta = schedule.beta(t)
        >>> beta.shape
        torch.Size([3])
    """

    def __init__(self, s: float = 0.008):
        self.s = s

    def alphas_cumprod(self, t: Tensor) -> Tensor:
        f = lambda u: torch.cos((u + self.s) / (1 + self.s) * np.pi / 2) ** 2
        device = t.device
        return f(t) / f(torch.zeros(1, device=device))

    def beta(self, t: Tensor) -> Tensor:
        return (np.pi / (2 * (1 + self.s))) * torch.tan((t + self.s) / (1 + self.s) * np.pi / 2)

    def integrated_beta(self, t: Tensor) -> Tensor:
        raise NotImplementedError("Cosine schedule does not use integrated beta explicitly.")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
