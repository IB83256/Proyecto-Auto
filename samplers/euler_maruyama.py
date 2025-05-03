# -*- coding: utf-8 -*-
"""
Euler-Maruyama integrator for stochastic differential equations (SDEs)

Author: Álvaro Duro y Carlos Beti
Date: [Fecha]
"""

from typing import Callable, Union
import torch
from torch import Tensor

def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,  
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    mask: Union[Tensor, None] = None,
    seed: Union[int, None] = None
) -> tuple[Tensor, Tensor]:
    """
    Euler-Maruyama integrator for SDEs of the form:
        dx_t = f(x_t, t) dt + g(t) dW_t

    Args:
        x_0: Initial tensor (e.g., images), shape (batch_size, n_channels, H, W)
        t_0: Initial time
        t_end: Final time
        n_steps: Number of discretization steps
        drift_coefficient: Function f(x, t) representing the drift term
        diffusion_coefficient: Function g(t) representing the diffusion term
        seed: Optional integer seed for reproducibility

    Returns:
        times: Tensor of discretized time points, shape (n_steps + 1,)
        x_t: Tensor containing the evolution of x over time,
             shape (*x_0.shape, n_steps + 1)

    Example:
        >>> import torch
        >>> x_0 = torch.zeros(2, 1, 4, 4)
        >>> drift = lambda x, t: torch.zeros_like(x)
        >>> diffusion = lambda t: torch.ones_like(t)
        >>> times, x_t = euler_maruyama_integrator(x_0, t_0=0.0, t_end=1.0, n_steps=5, drift_coefficient=drift, diffusion_coefficient=diffusion, seed=42)
        >>> x_t.shape
        torch.Size([2, 1, 4, 4, 6])
    """
    
    # Set reproducibility if seed is given
    if seed is not None:
        torch.manual_seed(seed)

    # Ensure same device and dtype as input
    device = x_0.device
    dtype = x_0.dtype

    # Create time vector
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device, dtype=dtype)
    dt = times[1] - times[0]
    dt_sqrt = torch.sqrt(torch.abs(dt))

    
    # Initialize x_t
    x_t = torch.empty((*x_0.shape, n_steps + 1), dtype=dtype, device=device)

    if mask is None:
        x_t[..., 0] = x_0
    else:
        # Inicializamos con ruido donde no se conoce el valor, y mantenemos x_0 donde sí
        x_init = torch.randn_like(x_0)
        x_init = x_init * (1 - mask) + x_0 * mask
        x_t[..., 0] = x_init


    # Sample all noise terms in advance
    z = torch.randn_like(x_t)

    # Integrate over time
    for n, t in enumerate(times[:-1]):
        t_tensor = torch.full((x_0.shape[0],), t.item(), device=device, dtype=dtype)

        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t_tensor) * dt
            + diffusion_coefficient(t_tensor).view(-1, 1, 1, 1)
              * dt_sqrt
              * z[..., n]
        )

    return times, x_t

if __name__ == "__main__":
    import doctest
    doctest.testmod()
