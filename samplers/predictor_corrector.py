# -*- coding: utf-8 -*-
"""
Predictor-Corrector sampler for score-based generative models.

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

from typing import Callable, Union
import torch
from torch import Tensor


def predictor_corrector_sampler(
    x_T: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    score_model: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    corrector_step_size: float = 0.01,
    n_corrector_steps: int = 1,
    seed: Union[int, None] = None,
) -> tuple[Tensor, Tensor]:
    """
    Predictor-Corrector integrator using Euler-Maruyama + Langevin correction.

    Args:
        x_T: Initial tensor sampled from prior, shape (B, C, H, W)
        t_0: Starting time (typically T=1.0)
        t_end: Ending time (typically close to 0)
        n_steps: Number of time discretization steps
        score_model: Callable s(x, t) that estimates the score ∇ log p_t(x)
        diffusion_coefficient: Function g(t)
        corrector_step_size: Langevin correction step size
        n_corrector_steps: Number of corrector steps per time step
        seed: Optional random seed

    Returns:
        times: Tensor of time points (n_steps + 1,)
        x_t: Tensor of image evolution over time, shape (B, C, H, W, n_steps + 1)
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = x_T.device
    dtype = x_T.dtype
    B = x_T.shape[0]

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]
    predictor_step_size = dt
    sqrt_corrector_step_size = torch.sqrt(torch.tensor(corrector_step_size, device=device, dtype=dtype))

    x_t = torch.empty((*x_T.shape, n_steps + 1), dtype=dtype, device=device)
    x_t[..., 0] = x_T

    for n, t in enumerate(times[:-1]):
        t_tensor = torch.full((B,), t.item(), device=device, dtype=dtype)

        # Predictor: Euler-Maruyama step
        g = diffusion_coefficient(t_tensor).view(-1, 1, 1, 1)
        z = torch.randn_like(x_t[..., n])
        drift = -g**2 * score_model(x_t[..., n], t_tensor)
        x_pred = (
            x_t[..., n]
            + drift * predictor_step_size
            + g * torch.sqrt(predictor_step_size) * z
        )

        # Corrector: Langevin dynamics
        x_corr = x_pred.clone()
        for _ in range(n_corrector_steps):
            noise = torch.randn_like(x_corr)
            score = score_model(x_corr, t_tensor)
            x_corr = x_corr + 0.5 * corrector_step_size * score + sqrt_corrector_step_size * noise

        x_t[..., n + 1] = x_corr

    return times, x_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()
