# base_controllers/chomp_slip_workspace/chomp_gradient_module/finite_difference_gradient.py

import numpy as np
from typing import Optional

from base_controllers.chomp_slip_workspace.chomp_gradient_module.base_gradient_module import (
    BaseGradientModule,
)


class FiniteDifferenceGradientModule(BaseGradientModule):
    """
    Standard central finite-difference gradient.

    Perturbs one coordinate at one internal waypoint at a time.
    """

    name = "finite_difference"

    def __init__(self, fd_eps: float = 1e-4, grad_clip: Optional[float] = None):
        self.fd_eps = fd_eps
        self.grad_clip = grad_clip

    def compute_gradient(self, cost_module, xi_xy, dt, dof, **kwargs):
        xi_xy = np.asarray(xi_xy, dtype=float)

        if xi_xy.ndim != 2:
            raise ValueError(f"xi_xy must be 2D, got shape {xi_xy.shape}")

        T, actual_dof = xi_xy.shape

        if actual_dof != dof:
            raise ValueError(
                f"Expected DOF={dof}, got xi_xy.shape[1]={actual_dof}"
            )

        if T < 3:
            raise ValueError(
                "Need at least 3 waypoints to compute internal gradient."
            )

        n_internal = T - 2
        grad = np.zeros((n_internal, dof), dtype=float)

        base_cost = cost_module.compute_cost(xi_xy=xi_xy, dt=dt, **kwargs)

        for t in range(1, T - 1):
            for d in range(dof):
                xi_plus = xi_xy.copy()
                xi_minus = xi_xy.copy()

                xi_plus[t, d] += self.fd_eps
                xi_minus[t, d] -= self.fd_eps

                cost_plus = cost_module.compute_cost(
                    xi_xy=xi_plus,
                    dt=dt,
                    **kwargs,
                )

                cost_minus = cost_module.compute_cost(
                    xi_xy=xi_minus,
                    dt=dt,
                    **kwargs,
                )

                grad[t - 1, d] = (
                    cost_plus - cost_minus
                ) / (2.0 * self.fd_eps)

        if self.grad_clip is not None:
            grad = np.clip(
                grad,
                -self.grad_clip,
                self.grad_clip,
            )

        grad_vec = grad.flatten(order="F")

        return grad_vec, base_cost