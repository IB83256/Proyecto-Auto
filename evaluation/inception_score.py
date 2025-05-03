import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from typing import Tuple
import numpy as np

def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    resize = transforms.Resize((299, 299))
    return resize(images)

def compute_inception_score(images: torch.Tensor,
                             device: str = "cuda",
                             splits: int = 10) -> Tuple[float, float]:
    """
    Calcula el Inception Score (IS) para un conjunto de imágenes generadas.

    Args:
        images: Tensor (N, C, H, W) con valores en [0, 1]
        device: 'cuda' o 'cpu'
        splits: número de particiones para estimar la varianza

    Returns:
        IS promedio y desviación estándar
    """
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    model.to(device)
    model.eval()

    images = preprocess_images(images)
    images = images.to(device)

    with torch.no_grad():
        preds = []
        for i in range(0, images.size(0), 32):
            batch = images[i:i+32]
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    N = preds.shape[0]
    scores = []

    for i in range(splits):
        part = preds[i * N // splits: (i + 1) * N // splits, :]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-8) - np.log(py[None, :]))
        kl_sum = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl_sum)))

    return float(np.mean(scores)), float(np.std(scores))

