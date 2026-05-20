import numpy as np
from typing import Optional
from base_controllers.chomp_slip_workspace.chomp_gradient_module.base_gradient_module import BaseGradientModule

class SPSAGradientModule(BaseGradientModule):
    """
    SPSA gradient approximation.

    Perturbs all internal waypoints simultaneously.
    Much cheaper than full finite differences for large trajectories.
    """

    name = "spsa"

    def __init__(
        self,
        perturbation_size: float = 1e-3,
        n_samples: int = 4,
        grad_clip: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        self.perturbation_size = perturbation_size
        self.n_samples = n_samples
        self.grad_clip = grad_clip
        self.rng = np.random.default_rng(random_seed)

    def compute_gradient(self, cost_module, xi_xy, dt, dof, **kwargs):
        xi_xy = np.asarray(xi_xy, dtype=float)

        if xi_xy.ndim != 2:
            raise ValueError(f"xi_xy must be 2D, got shape {xi_xy.shape}")

        T, actual_dof = xi_xy.shape

        if actual_dof != dof:
            raise ValueError(f"Expected DOF={dof}, got xi_xy.shape[1]={actual_dof}")

        if T < 3:
            raise ValueError("Need at least 3 waypoints to compute internal gradient.")

        n_internal = T - 2
        grad = np.zeros((n_internal, dof), dtype=float)

        base_cost = cost_module.compute_cost(xi_xy, dt, **kwargs)

        c = self.perturbation_size

        for _ in range(self.n_samples):
            # Random +/- 1 perturbation for internal waypoints only
            delta_internal = self.rng.choice(
                [-1.0, 1.0],
                size=(n_internal, dof),
            )

            delta_full = np.zeros_like(xi_xy)
            delta_full[1:-1, :] = delta_internal

            xi_plus = xi_xy + c * delta_full
            xi_minus = xi_xy - c * delta_full

            cost_plus = cost_module.compute_cost(xi_plus, dt, **kwargs)
            cost_minus = cost_module.compute_cost(xi_minus, dt, **kwargs)

            scalar_slope = (cost_plus - cost_minus) / (2.0 * c)

            # Since delta is +/-1, inverse(delta) = delta
            grad += scalar_slope * delta_internal

        grad /= float(self.n_samples)

        if self.grad_clip is not None:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        grad_vec = grad.flatten(order="F")

        return grad_vec, base_cost