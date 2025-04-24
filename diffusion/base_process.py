# -*- coding: utf-8 -*-
"""
Base class for diffusion processes.

Author: Álvaro Duro y Carlos Beti
"""

from typing import Callable
import torch
from torch import Tensor


class DiffusionProcess:
    def __init__(
        self,
        drift_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[Tensor], Tensor] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
