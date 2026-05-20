from abc import ABC, abstractmethod
import numpy as np


class BaseCostModule(ABC):
    """
    Base interface for all CHOMP-compatible cost modules.
    """

    name: str = "base_cost"

    @abstractmethod
    def compute_cost(self, xi_xy: np.ndarray, dt: float, **kwargs) -> float:
        """
        Compute scalar trajectory cost J(xi).
        """
        raise NotImplementedError