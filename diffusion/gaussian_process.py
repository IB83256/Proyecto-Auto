# -*- coding: utf-8 -*-
"""
Gaussian diffusion process.

Author: Álvaro Duro y Carlos Beti
"""

import torch
from torch import Tensor
from typing import Callable
from .base_process import DiffusionProcess

class GaussianDiffusionProcess(DiffusionProcess):
    """
    Gaussian Diffusion Process.

    Models both the forward SDE (adding noise) and the reverse-time SDE (sampling by removing noise).
    """
    def __init__(
        self,
        drift_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[Tensor], Tensor] = lambda t: 1.0,
        mu_t: Callable[[Tensor, Tensor], Tensor] = lambda x_0, t: x_0,
        sigma_t: Callable[[Tensor], Tensor] = lambda t: torch.sqrt(t),
    ):
        super().__init__(drift_coefficient, diffusion_coefficient)
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(
        self,
        score_model,
        x_0: Tensor,
        eps: float = 1.0e-5,
    ) -> Tensor:
        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=x_0.device) * (1.0 - eps) + eps
        z = torch.randn_like(x_0)
        sigma = self.sigma_t(t).view(-1, 1, 1, 1)
        x_t = x_0 + sigma * z
        score = score_model(x_t, t)
        loss = torch.mean(torch.norm((score * sigma + z), dim=(1, 2, 3), p=2) ** 2)
        return loss
    
    def reverse_drift(self, x_t: Tensor, t: Tensor, score_model) -> Tensor:
        """
        Computes the reverse-time drift:
            f_reverse(x, t) = f(x, t) - g(t)^2 * score_model(x, t)

        Args:
            x_t: Current state tensor.
            t: Current time tensor.
            score_model: Function (x_t, t) -> estimated score.

        Returns:
            Reverse drift tensor.
        """
        f_forward = self.drift_coefficient(x_t, t)
        g = self.diffusion_coefficient(t).view(-1, 1, 1, 1)  
        score = score_model(x_t, t)
        return f_forward - g**2 * score
