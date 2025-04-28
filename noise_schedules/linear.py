# -*- coding: utf-8 -*-
"""
Linear beta schedule for variance preserving diffusion models (OOP version).

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

from .base_schedule import NoiseSchedule
from torch import Tensor
import torch

class LinearSchedule(NoiseSchedule):
    """
    Linear beta schedule class for variance preserving diffusion models with general time interval [0, T].

    Args:
        beta_min: Minimum beta value.
        beta_max: Maximum beta value.
        T: Final time of the diffusion process (default: 1.0).

    Methods:
        alphas_cumprod(t): Computes cumulative alpha from t.
        integrated_beta(t): Computes the integral of beta from 0 to t.
        beta(t): Computes beta at time t.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def alphas_cumprod(self, t: Tensor) -> Tensor:
        return torch.exp(-self.integrated_beta(t))

    def integrated_beta(self, t: Tensor) -> Tensor:
        t_norm = t / self.T
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t_norm * t

    def beta(self, t: Tensor) -> Tensor:
        t_norm = t / self.T
        return self.beta_min + (self.beta_max - self.beta_min) * t_norm

if __name__ == "__main__":
    import doctest
    doctest.testmod()
