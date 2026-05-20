from base_controllers.chomp_slip_workspace.chomp_gradient_module.analytic_gradient import AnalyticGradientModule
from base_controllers.chomp_slip_workspace.chomp_gradient_module.finite_difference_gradient import FiniteDifferenceGradientModule
from base_controllers.chomp_slip_workspace.chomp_gradient_module.spsa_gradient import SPSAGradientModule

def make_gradient_module(gradient_name: str):
    if gradient_name == "finite_difference":
        return FiniteDifferenceGradientModule()

    if gradient_name == "spsa":
        return SPSAGradientModule()

    raise ValueError(f"Unknown gradient module: {gradient_name}")