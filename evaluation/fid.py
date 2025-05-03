import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from scipy.linalg import sqrtm
import numpy as np
from typing import Tuple

def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    """
    Prepara imágenes para InceptionV3: resize a 299x299, repite canal si necesario, escala a [-1,1].
    Args:
        images: Tensor de shape (N, C, H, W), con valores en [0,1]
    Returns:
        Tensor (N, 3, 299, 299)
    """
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # Grayscale -> RGB

    resize = transforms.Resize((299, 299))
    images = resize(images)
    images = images * 2 - 1  # Escala a [-1, 1] como espera Inception
    return images

def get_activations(images: torch.Tensor, model: nn.Module, batch_size: int = 32) -> torch.Tensor:
    """
    Pasa imágenes por InceptionV3 y devuelve los activations del penúltimo bloque (2048-d).
    Args:
        images: Tensor (N, 3, 299, 299)
        model: Modelo InceptionV3 sin la capa final
        batch_size: Tamaño del batch para inferencia
    Returns:
        Activaciones (N, 2048)
    """
    model.eval()
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(next(model.parameters()).device)
            pred = model(batch)
            activations.append(pred.cpu())
    return torch.cat(activations, dim=0)

def calculate_statistics(activations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula media y covarianza de las activaciones.
    """
    mu = activations.mean(dim=0).numpy()
    sigma = torch.from_numpy(np.cov(activations.numpy(), rowvar=False))
    return mu, sigma.numpy()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """
    Aplica la fórmula del FID entre dos distribuciones gaussianas.
    """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor, device: str = "cuda") -> float:
    """
    Calcula el Fréchet Inception Distance entre dos conjuntos de imágenes.
    Args:
        real_images: Tensor (N, C, H, W), valores en [0,1]
        generated_images: Tensor (N, C, H, W), valores en [0,1]
        device: 'cuda' o 'cpu'
    Returns:
        FID (float)
    """
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.to(device)

    real = preprocess_images(real_images)
    fake = preprocess_images(generated_images)

    act_real = get_activations(real, model)
    act_fake = get_activations(fake, model)

    mu_r, sigma_r = calculate_statistics(act_real)
    mu_g, sigma_g = calculate_statistics(act_fake)

    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

