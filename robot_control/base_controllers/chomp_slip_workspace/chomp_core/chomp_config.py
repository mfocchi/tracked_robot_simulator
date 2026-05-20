from dataclasses import dataclass


@dataclass
class ChompConfig:
    dof: int = 2
    n_knots: int = 40

    dt: float = 1.0
    max_iter: int = 100
    tol: float = 1.0

    eta: float = 0.001
    lambda_smooth: float = 200.0

    fd_eps: float = 1e-4
    grad_clip: float = 100.0

    save_history: bool = True