# -*- coding: utf-8 -*-
"""
Variance Preserving (VP) diffusion process.

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

import torch
from torch import Tensor
from .gaussian_process import GaussianDiffusionProcess
from noise_schedules.base_schedule import NoiseSchedule

class VPProcess(GaussianDiffusionProcess):
    """
    Variance Preserving (VP) diffusion process.

    Args:
        noise_schedule: Instance of a NoiseSchedule (e.g., LinearSchedule, CosineSchedule).

    Example:
        >>> from noise_schedules.linear import LinearSchedule
        >>> schedule = LinearSchedule(beta_min=0.1, beta_max=20.0)
        >>> process = VPProcess(noise_schedule=schedule)
        >>> x = torch.randn(2, 1, 4, 4)
        >>> t = torch.tensor([0.5, 0.5])
        >>> drift = process.drift_coefficient(x, t)
        >>> diffusion = process.diffusion_coefficient(t)
        >>> drift.shape
        torch.Size([2, 1, 4, 4])
        >>> diffusion.shape
        torch.Size([2])
    """

    def __init__(self, noise_schedule: NoiseSchedule):
        self.noise_schedule = noise_schedule

        drift_coefficient = lambda x_t, t: -0.5 * self.beta(t) * x_t
        diffusion_coefficient = lambda t: torch.sqrt(self.beta(t))

        def mu_t(x_0, t):
            return x_0 * torch.sqrt(self.noise_schedule.alphas_cumprod(t))

        def sigma_t(t):
            return torch.sqrt(1.0 - self.noise_schedule.alphas_cumprod(t))

        super().__init__(
            drift_coefficient=drift_coefficient,
            diffusion_coefficient=diffusion_coefficient,
            mu_t=mu_t,
            sigma_t=sigma_t,
        )

    def beta(self, t: Tensor) -> Tensor:
        """
        Beta(t) computed from the noise schedule.

        Returns:
            Beta(t) values.
        """
        return self.noise_schedule.beta(t)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
