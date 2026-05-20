import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve


class ChompOptimizer:
    """
    Modular CHOMP optimizer.

    This optimizer does not know which task cost is used.
    It does not know which gradient method is used.

    It combines:

        total_cost =
            lambda_smooth * smoothness_cost
            + task_cost

        total_gradient =
            lambda_smooth * smoothness_gradient
            + task_gradient

    and applies the CHOMP/covariant update:

        delta_xi = -A^{-1} total_gradient

    where A is the smoothness Hessian matrix.
    """

    def __init__(self, config, cost_module, gradient_module):
        self.config = config
        self.cost_module = cost_module
        self.gradient_module = gradient_module

        # Optional parameters.
        # If they are not inside ChompConfig, defaults are used.
        self.use_backtracking = getattr(config, "use_backtracking", True)
        self.backtracking_start_iter = getattr(config, "backtracking_start_iter", 5)
        self.backtracking_beta = getattr(config, "backtracking_beta", 0.5)
        self.min_eta = getattr(config, "min_eta", 1e-5)

    # ==================================================
    # Public optimization function
    # ==================================================

    def optimize(self, xi0):
        """
        Optimize a trajectory.

        Parameters
        ----------
        xi0 : ndarray, shape (T, DOF)
            Initial trajectory in CHOMP/world coordinates.

        Returns
        -------
        xi : ndarray, shape (T, DOF)
            Optimized trajectory.

        cost_history : list[float]
            Total cost history.

        trajectory_history : list[ndarray]
            Saved trajectory history.
        """

        xi = np.asarray(xi0, dtype=float).copy()

        T = xi.shape[0]
        dof = self.config.dof
        dt = self.config.dt

        if xi.ndim != 2:
            raise ValueError("xi0 must have shape (T, DOF).")

        if xi.shape[1] != dof:
            raise ValueError(
                f"Trajectory DOF mismatch: xi has {xi.shape[1]} columns, "
                f"but config.dof={dof}."
            )

        if T < 3:
            raise ValueError("CHOMP requires at least 3 knots.")

        # --------------------------------------------------
        # Precompute smoothness matrix structure
        # --------------------------------------------------
        A, Q_full = self.build_smoothness_matrix(
            T=T,
            dof=dof,
        )

        cost_history = []
        task_cost_history = []
        smoothness_cost_history = []
        trajectory_history = []

        # ==================================================
        # Optimization loop
        # ==================================================

        for iteration in range(1, self.config.max_iter + 1):

            # --------------------------------------------------
            # 1) Compute task cost and task gradient
            # --------------------------------------------------
            task_cost = self.compute_task_cost(
                xi=xi,
                dt=dt,
            )

            task_grad_vec, _ = self.gradient_module.compute_gradient(
                cost_module=self.cost_module,
                xi_xy=xi,
                dt=dt,
                dof=dof,
            )

            task_grad_vec = np.asarray(task_grad_vec, dtype=float).reshape(-1)

            # --------------------------------------------------
            # 2) Compute smoothness cost and smoothness gradient
            # --------------------------------------------------
            smoothness_cost, smoothness_grad_vec = self.compute_smoothness_cost_and_gradient(
                xi=xi,
                Q_full=Q_full,
                dof=dof,
            )

            # --------------------------------------------------
            # 3) Combine task and smoothness terms
            # --------------------------------------------------
            total_cost = (
                self.config.lambda_smooth * smoothness_cost
                + task_cost
            )

            total_grad_vec = (
                self.config.lambda_smooth * smoothness_grad_vec
                + task_grad_vec
            )

            # --------------------------------------------------
            # 4) Save history
            # --------------------------------------------------
            if self.config.save_history:
                trajectory_history.append(xi.copy())

            cost_history.append(total_cost)
            task_cost_history.append(task_cost)
            smoothness_cost_history.append(smoothness_cost)

            # --------------------------------------------------
            # 5) CHOMP covariant update
            # --------------------------------------------------
            delta_vec = -spsolve(A, total_grad_vec)

            delta_int = self.vector_to_internal_waypoints(
                delta_vec=delta_vec,
                T=T,
                dof=dof,
            )

            step_norm = np.linalg.norm(self.config.eta * delta_int)
            grad_norm = np.linalg.norm(total_grad_vec)

            print(
                f"Iteration {iteration:3d} | "
                f"total cost: {total_cost:12.6f} | "
                f"task cost: {task_cost:12.6f} | "
                f"smooth cost: {smoothness_cost:12.6f} | "
                f"grad norm: {grad_norm:12.6f} | "
                f"step norm: {step_norm:12.6f}"
            )

            # --------------------------------------------------
            # 6) Stopping criterion
            # --------------------------------------------------
            if step_norm < self.config.tol:
                print(
                    f"Converged at iteration {iteration}: "
                    f"step norm {step_norm:.6f} < tol {self.config.tol:.6f}"
                )
                break

            # --------------------------------------------------
            # 7) Apply update, optionally with backtracking
            # --------------------------------------------------
            if (
                self.use_backtracking
                and iteration >= self.backtracking_start_iter
            ):
                xi = self.apply_update_with_backtracking(
                    xi=xi,
                    delta_int=delta_int,
                    current_total_cost=total_cost,
                    Q_full=Q_full,
                    dof=dof,
                    dt=dt,
                )
            else:
                xi[1:-1, :] += self.config.eta * delta_int

        return xi, cost_history, trajectory_history

    # ==================================================
    # Task cost wrapper
    # ==================================================

    def compute_task_cost(self, xi, dt):
        """
        Compute task cost through the selected cost module.
        """

        return self.cost_module.compute_cost(
            xi_xy=xi,
            dt=dt,
        )

    # ==================================================
    # Smoothness term
    # ==================================================

    def build_smoothness_matrix(self, T, dof):
        """
        Build the CHOMP smoothness Hessian matrix.

        The smoothness cost is based on squared acceleration:

            F_smooth = 1/2 sum_i || xi[i-1] - 2 xi[i] + xi[i+1] ||^2

        Only internal waypoints are optimized.

        Returns
        -------
        A : sparse matrix, shape (dof * (T - 2), dof * (T - 2))
            Internal smoothness Hessian.

        Q_full : sparse matrix, shape (T, T)
            Full 1D smoothness Hessian before removing fixed endpoints.
        """

        Nint = T - 2

        # --------------------------------------------------
        # Second-difference matrix D
        # D xi = xi[i] - 2 xi[i+1] + xi[i+2]
        # --------------------------------------------------
        diagonals = [
            np.ones(T - 2),
            -2.0 * np.ones(T - 2),
            np.ones(T - 2),
        ]

        offsets = [0, 1, 2]

        D = diags(
            diagonals,
            offsets,
            shape=(T - 2, T),
            format="csc",
        )

        # Full Hessian for one coordinate.
        Q_full = D.T @ D

        # Internal block, because endpoints are fixed.
        Q_int = Q_full[1:-1, 1:-1]

        # For column-major waypoint vector:
        #
        #     [x_1, x_2, ..., x_Nint, y_1, y_2, ..., y_Nint]
        #
        # the full Hessian is kron(I_dof, Q_int).
        A = kron(
            eye(dof, format="csc"),
            Q_int,
            format="csc",
        )

        return csc_matrix(A), csc_matrix(Q_full)

    def compute_smoothness_cost_and_gradient(self, xi, Q_full, dof):
        """
        Compute smoothness cost and gradient for internal waypoints.

        Parameters
        ----------
        xi : ndarray, shape (T, DOF)
            Full trajectory, including fixed endpoints.

        Q_full : sparse matrix, shape (T, T)
            Full smoothness Hessian for one coordinate.

        dof : int
            Degrees of freedom.

        Returns
        -------
        smoothness_cost : float

        smoothness_grad_vec : ndarray, shape (dof * (T - 2),)
            Gradient with respect to internal waypoints only,
            flattened in column-major order.
        """

        T = xi.shape[0]

        smoothness_cost = 0.0
        grad_int = np.zeros((T - 2, dof), dtype=float)

        for d in range(dof):
            q = xi[:, d]

            smoothness_cost += 0.5 * float(q.T @ (Q_full @ q))

            grad_full = Q_full @ q
            grad_int[:, d] = grad_full[1:-1]

        smoothness_grad_vec = grad_int.flatten(order="F")

        return smoothness_cost, smoothness_grad_vec

    # ==================================================
    # Vector reshaping utilities
    # ==================================================

    def vector_to_internal_waypoints(self, delta_vec, T, dof):
        """
        Convert a CHOMP vector into internal waypoint matrix form.

        Input order is column-major:

            [x_1, x_2, ..., x_Nint, y_1, y_2, ..., y_Nint]

        Output shape:

            (T - 2, DOF)
        """

        return delta_vec.reshape(
            (T - 2, dof),
            order="F",
        )

    # ==================================================
    # Backtracking
    # ==================================================

    def apply_update_with_backtracking(
        self,
        xi,
        delta_int,
        current_total_cost,
        Q_full,
        dof,
        dt,
    ):
        """
        Apply update with simple backtracking line search.

        The update is accepted if the new total cost is not larger than the
        current total cost.
        """

        eta_try = self.config.eta

        while True:
            xi_try = xi.copy()
            xi_try[1:-1, :] += eta_try * delta_int

            task_cost_try = self.compute_task_cost(
                xi=xi_try,
                dt=dt,
            )

            smoothness_cost_try, _ = self.compute_smoothness_cost_and_gradient(
                xi=xi_try,
                Q_full=Q_full,
                dof=dof,
            )

            total_cost_try = (
                self.config.lambda_smooth * smoothness_cost_try
                + task_cost_try
            )

            if total_cost_try <= current_total_cost:
                print(
                    f"    accepted eta = {eta_try:.6e}, "
                    f"new cost = {total_cost_try:.6f}"
                )
                return xi_try

            if eta_try < self.min_eta:
                print(
                    f"    backtracking stopped at eta = {eta_try:.6e}, "
                    f"cost did not improve"
                )
                return xi_try

            eta_try *= self.backtracking_beta