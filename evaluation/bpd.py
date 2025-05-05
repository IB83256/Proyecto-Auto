import torch
import torch.nn.functional as F
from typing import Callable

def compute_bpd(x_0: torch.Tensor,
                score_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                diffusion_process,
                eps: float = 1e-5) -> float:
    """
    Calcula los bits por dimensión (BPD) para evaluar la log-verosimilitud aproximada.

    Args:
        x_0: Tensor de imágenes originales (batch_size, C, H, W), valores en [0, 1].
        score_model: Red que estima el score \nabla log p(x_t | t)
        diffusion_process: Objeto con métodos sigma_t(t), mu_t(x_0, t) y loss_function()
        eps: Valor mínimo de tiempo para evitar inestabilidades numéricas

    Returns:
        BPD promedio sobre el batch (float)
    """
    batch_size = x_0.shape[0]

    # (1) Muestreamos t ~ Uniform(eps, 1)
    t = torch.rand(batch_size, device=x_0.device) * (1.0 - eps) + eps

    # (2) Muestreamos ruido z ~ N(0, I)
    z = torch.randn_like(x_0)

    # (3) Calculamos x_t = mu(t) + sigma(t) * z
    sigma = diffusion_process.sigma_t(t).view(-1, 1, 1, 1)
    mu = diffusion_process.mu_t(x_0, t)
    x_t = mu + sigma * z

    # (4) Score predicho por la red
    score = score_model(x_t, t)

    # (5) Estimador de logp(x_0) basado en el score-matching
    loss = torch.mean((score * sigma + z).pow(2).sum(dim=(1, 2, 3)))

    # (6) Bits por dimensión = logp / (ln(2) * n_dim)
    n_dim = x_0[0].numel()
    bpd = loss * 0.5 * (1.0 / torch.log(torch.tensor(2.0))) / n_dim

    return bpd.item()
