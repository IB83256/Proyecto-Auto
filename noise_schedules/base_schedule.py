from abc import ABC, abstractmethod
from torch import Tensor

class NoiseSchedule(ABC):
    @abstractmethod
    def alphas_cumprod(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def integrated_beta(self, t: Tensor) -> Tensor:
        pass
