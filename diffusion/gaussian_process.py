# -*- coding: utf-8 -*-
"""
Gaussian diffusion process.

Author: Álvaro Duro y Carlos Beti
"""

import torch
from torch import Tensor
from typing import Callable
import numpy as np
from .base_process import DiffusionProcess


class GaussianDiffusionProcess(DiffusionProcess):
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
