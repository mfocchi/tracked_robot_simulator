from abc import ABC, abstractmethod
import numpy as np

from base_controllers.chomp_slip_workspace.chomp_cost_module.base_cost_module import BaseCostModule


class BaseGradientModule(ABC):
    """
    Base interface for gradient computation modules.

    A gradient module does not define the cost.
    It only defines how to compute/approximate the gradient of a cost.
    """

    name: str = "base_gradient"

    @abstractmethod
    def compute_gradient(
        self,
        cost_module: BaseCostModule,
        xi_xy: np.ndarray,
        dt: float,
        dof: int,
        **kwargs,
    ):
        """
        Compute gradient of J(xi).

        Returns
        -------
        grad_vec:
            Flattened gradient over internal waypoints.
        cost:
            Scalar cost value.
        """
        raise NotImplementedError