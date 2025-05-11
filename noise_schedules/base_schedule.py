# -*- coding: utf-8 -*-
"""
Abstract base class for noise schedules used in diffusion processes.

Defines the interface for computing cumulative alpha products and integrated beta functions 
across different types of schedules (e.g., linear, cosine).

Author: Álvaro Duro y Carlos Beti  
Date: 2025-05-01
"""

from abc import ABC, abstractmethod
from torch import Tensor

class NoiseSchedule(ABC):
    @abstractmethod
    def alphas_cumprod(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def integrated_beta(self, t: Tensor) -> Tensor:
        pass
