from base_controllers.chomp_slip_workspace.chomp_gradient_module.base_gradient_module import BaseGradientModule


class AnalyticGradientModule(BaseGradientModule):
    """
    Calls cost_module.compute_analytic_gradient(...).

    Use this only for cost modules that implement analytic gradients.

    For example we can base our perturbation based on the terrain geometry
    """

    name = "analytic"

    def compute_gradient(self, cost_module, xi_xy, dt, dof, **kwargs):
        if not hasattr(cost_module, "compute_analytic_gradient"):
            raise TypeError(
                f"{cost_module.__class__.__name__} does not implement "
                "compute_analytic_gradient(...)."
            )

        return cost_module.compute_analytic_gradient(
            xi_xy=xi_xy,
            dt=dt,
            dof=dof,
            **kwargs,
        )