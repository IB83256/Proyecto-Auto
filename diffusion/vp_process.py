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

        drift_coefficient = lambda x_t, t: -0.5 * self.beta(t).view(-1, 1, 1, 1) * x_t
        diffusion_coefficient = lambda t: torch.sqrt(self.beta(t))

        def mu_t(x_0, t):
            return x_0 * torch.sqrt(self.noise_schedule.alphas_cumprod(t))

        def sigma_t(t):
            return torch.sqrt(1.0 - self.noise_schedule.alphas_cumprod(t))
        
        def sigma_t_integrated(t):
            with torch.no_grad():
                num_points = 500
                sigmas = []
                for t_i in t:
                    s_vals = torch.linspace(0, t_i.item(), num_points, device=t.device)
                    ds = t_i.item() / (num_points - 1)
                    beta_s = self.beta(s_vals)

                    try:
                        int_beta_t = self.noise_schedule.integrated_beta(torch.tensor([t_i], device=t.device))[0]
                        int_beta_s = self.noise_schedule.integrated_beta(s_vals)
                        inner_integrals = int_beta_t - int_beta_s
                    except NotImplementedError:
                        inner_integrals = []
                        for s_j in s_vals:
                            u_vals = torch.linspace(s_j.item(), t_i.item(), num_points, device=t.device)
                            du = (t_i.item() - s_j.item()) / (num_points - 1)
                            beta_u = self.beta(u_vals)
                            int_beta = torch.trapz(beta_u, dx=du)
                            inner_integrals.append(int_beta.item())
                        inner_integrals = torch.tensor(inner_integrals, device=t.device)

                    integrand = torch.exp(-0.5 * inner_integrals) * beta_s
                    sigma2 = torch.trapz(integrand, dx=ds)
                    sigmas.append(torch.sqrt(sigma2))
                return torch.stack(sigmas)

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
