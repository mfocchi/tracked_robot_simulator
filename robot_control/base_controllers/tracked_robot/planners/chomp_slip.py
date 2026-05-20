import numpy as np
import matplotlib

try:
    matplotlib.use("QtAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from base_controllers.tracked_robot.planners.chomp_no_theta import Params, ChompSolver
from base_controllers.evaluate_energy_consumption import EvaluateEnergyConsumption
from scipy.sparse.linalg import spsolve

robotName = "tractor" # needs to inherit BaseController

class ChompSolverSlip(ChompSolver):
    def __init__(self, task_name='slip'):
        super().__init__(task_name=task_name)

        self.evaluateEnergyConsumption = EvaluateEnergyConsumption(dt=0.004)
        self.evaluateEnergyConsumption.DEBUG = True
        self.fd_eps = 1e-3   # finite-difference perturbation in world units
        self.grad_clip = 1e3 # optional stability clip

        self.sx = 1.0
        self.sy = 1.0

    def addThetaFromXY(self, xi_xy, theta0=None):
        """
        Build a full SE(2) trajectory (T x 3) from an XY-only trajectory (T x 2)
        by setting theta to the local tangent direction of the curve.

        Endpoints use one-sided differences, internal points use centered differences.
        """
        xi_xy = np.asarray(xi_xy, dtype=float)
        T = xi_xy.shape[0]
        assert xi_xy.shape[1] == 2

        x = xi_xy[:, 0]
        y = xi_xy[:, 1]
        theta = np.zeros(T, dtype=float)

        if T >= 2:
            theta[0] = np.arctan2(y[1] - y[0], x[1] - x[0])
            theta[-1] = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])

        for t in range(1, T - 1):
            dx = x[t + 1] - x[t - 1]
            dy = y[t + 1] - y[t - 1]
            theta[t] = np.arctan2(dy, dx)

        if theta0 is not None:
            theta[0] = float(theta0)

        theta = np.unwrap(theta)
        xi_full = np.column_stack([x, y, theta])
        return xi_full

    # ------------------------------------------------------------------
    # Trajectory -> motion quantities needed by energy evaluator
    # ------------------------------------------------------------------
    def trajectory_to_motion(self, xi_xy, dt):
        """
        Converts XY trajectory in CHOMP/world units into:
          - x [m]
          - y [m]
          - theta [rad]
          - v [m/s]
          - omega [rad/s]

        Returns:
            des_x, des_y, des_theta, v_des, omega_des
        """
        xi_xy = np.asarray(xi_xy, dtype=float)

        # Map to meters
        xi_meters = xi_xy.copy()
        xi_meters[:, 0] *= self.sx
        xi_meters[:, 1] *= self.sy

        # Reconstruct theta from tangent in metric space
        xi_full_meters = self.addThetaFromXY(xi_meters)
        des_x = xi_full_meters[:, 0]
        des_y = xi_full_meters[:, 1]
        des_theta = xi_full_meters[:, 2]

        # Finite differences for v and omega
        dx = np.diff(des_x)
        dy = np.diff(des_y)
        dtheta = np.diff(np.unwrap(des_theta))

        # Keep same length as the trajectory
        if len(dx) == 0:
            v_des = np.zeros_like(des_x)
            omega_des = np.zeros_like(des_theta)
        else:
            dx = np.append(dx, dx[-1])
            dy = np.append(dy, dy[-1])
            dtheta = np.append(dtheta, dtheta[-1])

            v_des = np.hypot(dx, dy) / dt
            omega_des = dtheta / dt

        return des_x, des_y, des_theta, v_des, omega_des


    # Energy cost that returns a scalar and not a vector
    def energy_cost(self, xi_xy, dt):
        """
        Black-box scalar energy functional for CHOMP:
            F_energy(xi) = sum(computeCost(...))

        computeCost() returns a vector of per-segment/per-knot costs.
        CHOMP needs one scalar objective, so we sum the vector here.
        """

        des_x, des_y, des_theta, v_des, omega_des = self.trajectory_to_motion(
            xi_xy,
            dt
        )

        cost_vec = self.evaluateEnergyConsumption.computeCost(
            des_x_vec=des_x,
            des_y_vec=des_y,
            des_yaw_vec=des_theta,
            v_ol=v_des,
            omega_ol=omega_des,
            plan_dt=dt
        )

        cost_vec = np.asarray(cost_vec, dtype=float)

        if cost_vec.size == 0:
            return 0.0

        if not np.all(np.isfinite(cost_vec)):
            print("Warning: energy cost vector contains NaN or inf.")
            cost_vec = np.nan_to_num(
                cost_vec,
                nan=0.0,
                posinf=1e12,
                neginf=-1e12
            )

        return float(np.sum(cost_vec))

    # ------------------------------------------------------------------
    # Energy functional gradient by finite differences
    # ------------------------------------------------------------------
    def compute_slip_cost_gradient(self, xi_xy, robot, M, dt, DOF):  # Correct but slow calculation
        """
        Compute:
            nabla_Fenergy_vec, Fenergy_cost

        Inputs:
          xi_xy : (T x 2) trajectory in CHOMP/world units
          robot, M : unused directly here, kept for CHOMP interface compatibility
          dt : timestep
          DOF : should be 2 in this formulation

        Outputs:
          nabla_Fenergy_vec : length DOF*(T-2), column-major stacked
          Fenergy_cost      : scalar
        """
        T = xi_xy.shape[0]
        Nint = T - 2

        if DOF != 2:
            raise ValueError(f"ChompSolverSlip assumes DOF=2, got DOF={DOF}")

        grad = np.zeros((Nint, DOF), dtype=float)

        # Base cost
        Fenergy_cost = self.energy_cost(xi_xy, dt)

        # Finite-difference gradient wrt internal XY waypoints only
        for t in range(1, T - 1):
            for d in range(DOF):
                xi_plus = xi_xy.copy()
                xi_minus = xi_xy.copy()

                xi_plus[t, d] += self.fd_eps
                xi_minus[t, d] -= self.fd_eps

                cost_plus = self.energy_cost(xi_plus, dt)
                cost_minus = self.energy_cost(xi_minus, dt)

                grad[t - 1, d] = (cost_plus - cost_minus) / (2.0 * self.fd_eps)

        # Optional clipping for numerical stability
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        # Flatten in MATLAB / CHOMP column-major style
        nabla_Fenergy_vec = grad.flatten(order='F')


        return nabla_Fenergy_vec, Fenergy_cost

    def compute_slip_cost_gradient_light(self, xi_xy, robot, M, dt, DOF):   # Not correct but fast
        """
        Compute:
            nabla_Fenergy_vec, Fenergy_cost

        Inputs:
          xi_xy : (T x 2) trajectory in CHOMP/world units
          robot, M : unused directly here, kept for CHOMP interface compatibility
          dt : timestep
          DOF : should be 2 in this formulation

        Outputs:
          nabla_Fenergy_vec : length DOF*(T-2), column-major stacked
          Fenergy_cost      : scalar
        """
        T = xi_xy.shape[0]
        Nint = T - 2

        if DOF != 2:
            raise ValueError(f"ChompSolverSlip assumes DOF=2, got DOF={DOF}")

        EnergyVector = self.energy_cost(xi_xy, dt)
        # Base cost
        Fenergy_cost = np.sum(EnergyVector)

        # Finite-difference gradient wrt internal XY waypoints only
        # grad along X
        x_plus = xi_xy.copy()
        x_minus = xi_xy.copy()

        # the whole trajectory is shifted with the same x component
        x_plus[:,0] += self.fd_eps
        x_minus[:,0] -= self.fd_eps

        cost_plus_x = self.energy_cost(x_plus, dt)
        cost_minus_x = self.energy_cost(x_minus, dt)
        grad_x = (cost_plus_x - cost_minus_x) / (2.0 * self.fd_eps)

        #grad along Y
        y_plus = xi_xy.copy()
        y_minus = xi_xy.copy()

        # the whole trajectory is shifted with the same y component
        y_plus[:,1] += self.fd_eps
        y_minus[:,1] -= self.fd_eps

        cost_plus_y = self.energy_cost(y_plus, dt)
        cost_minus_y = self.energy_cost(y_minus, dt)
        grad_y = (cost_plus_y - cost_minus_y) / (2.0 * self.fd_eps)

        #this is a vector N-2 X 2
        grad = np.hstack((grad_x[1:T-1], grad_y[1:T-1]))

        # Optional clipping for numerical stability
        #grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        # Flatten in MATLAB / CHOMP column-major style
        nabla_Fenergy_vec = grad.flatten(order='F') # 1D array length of dimension 2(N-2)

        print(f"cost for slippage is {Fenergy_cost}, gradient for slippage is {nabla_Fenergy_vec} ")

        return nabla_Fenergy_vec, Fenergy_cost

    # Override the calculate_total_gradient function of the ChompSolver in chomp_no_theta using the compute_slip_cost_gradient function
    def calculate_total_gradient(self, xi, T, robot, params, M, A, b_vec, c, lambda_, DOF):

        dT = params.dT
        Nint = T - 2

        # Internal points only
        xi_int = xi[1:-1, :]  # (Nint x 2)
        xi_int_vec = xi_int.T.reshape(-1)  # DOF-major, compatible with A

        # Smoothness term
        Fsmooth_cost = 0.5 * (xi_int_vec.T @ (A @ xi_int_vec)) + np.dot(b_vec, xi_int_vec) + c
        nabla_Fsmooth_vec = (A @ xi_int_vec) + b_vec

        # Energy term
        nabla_Ftask_vec, Ftask_cost = self.compute_slip_cost_gradient(xi, robot, M, dT, DOF)    # use the slow one

        # Total functional
        nabla_U_vec = lambda_ * nabla_Fsmooth_vec + nabla_Ftask_vec
        total_cost = lambda_ * Fsmooth_cost + Ftask_cost

        return (
            nabla_U_vec,
            nabla_Ftask_vec,
            nabla_Fsmooth_vec,
            total_cost,
            Ftask_cost,
            Fsmooth_cost
        )

    # ChompSolve with saving the chomp history
    def chomp_solve_with_history(self, params, robot, M, xi, h_image_metric=None):
        """
        finalTrajectory = chompSolver(params, robot, M, xi)
        """

        # History of the paths (to put into the GIF)
        chomp_history = [xi.copy()]

        DOF = params.DOF
        lambda_ = params.lambda_
        MAX_ITER = params.MAX_ITER
        TOL = params.TOL

        T = xi.shape[0]

        xi_int = xi[1:T - 1, :]
        N_int = T - 2  # Number of free waypoints

        # Precompute A, b, c (smoothness) The Dynamics (Hessian) Matrix A
        A, b_vec, c = self.calculate_A_matrix(T, DOF, xi)

        print("Starting CHOMP optimization...")
        # --- plotting setup ---
        h_traj = None
        title_handle = None
        if h_image_metric is not None:
            plt.figure(h_image_metric.number)
            # create a single line object for the trajectory
            (h_traj,) = plt.plot(xi[:, 0], xi[:, 1], "bo-", linewidth=2, markersize=5, label="Current traj")
            title_handle = plt.title("Iteration 0")
            plt.legend()
            plt.draw()
            plt.pause(0.01)
        # --------------------

        for it in range(1, MAX_ITER + 1):
            print(f"\nIteration {it:3d} |", end="")

            # Step 1: total functional gradient
            (nabla_U_vec, nabla_Ftask_vec, nabla_Fsmooth_vec, total_cost, Ftask_cost, Fsmooth_cost) \
                = self.calculate_total_gradient(xi, T, robot, params, M, A, b_vec, c, lambda_, DOF)

            # todo check  nabla_Ftask_vec is twice higher than the matlab one

            # Step 2: covariant update: delta_xi = -A^{-1} * nabla_U
            delta_xi_vec = -spsolve(A, nabla_U_vec)

            # reshape back to internal (T-2 x DOF)
            delta_xi_int = delta_xi_vec.reshape((DOF, N_int)).T  # (Nint x DOF)

            # init eta to default value
            eta = params.eta

            if it < 5:
                # without backtracking
                xi_int_new = xi_int + (eta) * delta_xi_int
            else:
                # with backtracking
                # Right after computing xi_int_new, compute cost at the candidate and reject if worse (basic backtracking):
                eta_try = eta
                while True:
                    xi_try_int = xi_int + eta_try * delta_xi_int

                    # build full trajectory xi_try with fixed endpoints
                    xi_try = np.zeros_like(xi)
                    xi_try[0, :] = xi[0, :]
                    xi_try[T - 1, :] = xi[T - 1, :]
                    xi_try[1:T - 1, :] = xi_try_int

                    (nabla_U_vec_try,
                     nabla_Ftask_vec_try,
                     nabla_Fsmooth_try,
                     total_cost_try,
                     Ftask_cost_try,
                     Fsmooth_cost_try) = self.calculate_total_gradient(
                        xi_try, T, robot, params, M, A, b_vec, c, lambda_, DOF
                    )

                    if (total_cost_try <= total_cost) or (eta_try < 1e-5):
                        xi_int_new = xi_try_int
                        eta = eta_try
                        break

                    eta_try = eta_try * 0.5
                    # print(f"\n        prev cost {total_cost:6.0f} |      new_cost {total_cost_try:6.0f}, backtracking updating  eta to {eta_try:f} \n", end="")

            # --- Step 5: Update Trajectory ---
            xi_int = xi_int_new

            # Reconstruct full trajectory for the next iteration (fixed boundaries)
            xi[1:T - 1, :] = xi_int
            chomp_history.append(xi.copy())

            # Logging
            grad_s_norm = np.linalg.norm(lambda_ * nabla_Fsmooth_vec)
            grad_o_norm = np.linalg.norm(nabla_Ftask_vec)
            step_norm = np.linalg.norm(eta * delta_xi_int)

            print(f"   total Cost: {total_cost:8.0f} |"
                  f"    Smoothness cost: {lambda_ * Fsmooth_cost:8.0f} |"
                  f"    Slip cost: {Ftask_cost:8.0f} |"
                  f"    Smooth Grad.norm: {grad_s_norm:8.0f} |"
                  f"    Slip Grad.norm: {grad_o_norm:8.0f} |"
                  f" Step Norm: {step_norm:8.0f}", end="")

            if step_norm < TOL:
                print(f"\nConverged after {it} iterations because step size is below TOL.")
                break

            # --- UPDATE PLOT HERE ---
            if h_image_metric is not None and h_traj is not None:
                h_traj.set_xdata(xi[:, 0])
                h_traj.set_ydata(xi[:, 1])

                if title_handle is not None:
                    title_handle.set_text(f"Iteration {it}")

                plt.draw()
                plt.pause(0.1)

        xi_full = self.addThetaFromXY(xi)
        return xi_full, chomp_history

    # ------------------------------------------------------------------
    # Optional: custom optimize wrapper for plotting labels
    # ------------------------------------------------------------------
    def optimize_with_history(self, xi0, M, params, robot):
        plt.ion()
        h_image_metric = self.plotMap(M)

        plt.figure(h_image_metric.number)
        plt.plot(xi0[:, 0], xi0[:, 1], "ko-", linewidth=2, label="Initial")

        xi_full, chomp_history = self.chomp_solve_with_history(
            params,
            robot,
            M,
            xi0,
            h_image_metric=h_image_metric
        )

        plt.figure(h_image_metric.number)
        plt.plot(
            xi_full[:, 0],
            xi_full[:, 1],
            "ro-",
            linewidth=2,
            label="CHOMP-energy"
        )

        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        return xi_full, chomp_history


if __name__ == "__main__":

    # --------------------------------------------------
    # 1) Create CHOMP solver and terrain evaluator
    # --------------------------------------------------
    ch = ChompSolverSlip()
    evaluator = EvaluateEnergyConsumption()

    # --------------------------------------------------
    # 2) Compute terrain height grid
    # --------------------------------------------------
    X, Y, Z, x_edges, y_edges = evaluator.computeTerrainHeightGrid(
        nx=150,
        ny=150,
        samples_per_cell=1,
        z_margin=5.0
    )

    evaluator.plotTerrainHeightGrid(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map.png",
        show=False
    )

    evaluator.printTerrainHeightCandidates(X, Y, Z, n=10)

    # --------------------------------------------------
    # 3) Create CHOMP map without obstacles
    # --------------------------------------------------
    obstacles = []

    xRange = np.array([0.0, 500.0])
    yRange = np.array([0.0, 500.0])

    rows = 2000
    cols = 2000
    epsilon = 50.0

    M = ch.constructMap(
        xRange,
        yRange,
        rows,
        cols,
        obstacles,
        epsilon
    )

    # --------------------------------------------------
    # 4) Scaling: world units -> meters
    # --------------------------------------------------
    xL_world = xRange[1] - xRange[0]
    yL_world = yRange[1] - yRange[0]

    xL_m_des = 10.0
    yL_m_des = 10.0

    ch.sx = xL_m_des / xL_world
    ch.sy = yL_m_des / yL_world

    # --------------------------------------------------
    # 5) Start, goal, CHOMP parameters
    # --------------------------------------------------
    theta0 = np.pi / 4.0

    q_start = np.array([0.0, 0.0, theta0])
    q_goal = np.array([300.0, 200.0, theta0])

    chomp_params = Params(
        DOF=2,
        lambda_=200.0,
        eta=0.001,
        MAX_ITER=100,
        TOL=1.0,
        dT=1,
        t0=0.0,
        tf=40.0,
        convex_hull_contact=True,
    )

    # --------------------------------------------------
    # 6) Robot footprint
    # --------------------------------------------------
    w = 60.0
    h = 40.0

    X_robot = np.array(
        [-w / 2, w / 2, w / 2, -w / 2],
        dtype=float
    )

    Y_robot = np.array(
        [-h / 2, -h / 2, h / 2, h / 2],
        dtype=float
    )

    robot = ch.createRobot(
        X_robot,
        Y_robot,
        q_start,
        M,
        chomp_params.convex_hull_contact
    )

    # --------------------------------------------------
    # 7) Initial straight-line trajectory in world units
    # --------------------------------------------------
    T = int(chomp_params.tf / chomp_params.dT)

    xi0 = np.zeros((T, 2), dtype=float)

    xi0[:, 0] = np.linspace(q_start[0], q_goal[0], T)
    xi0[:, 1] = np.linspace(q_start[1], q_goal[1], T)

    # --------------------------------------------------
    # 8) Run CHOMP-slip optimization and collect history
    # --------------------------------------------------
    xi_full, chomp_history = ch.optimize_with_history(
        xi0,
        M,
        chomp_params,
        robot
    )

    # --------------------------------------------------
    # 9) Convert history from world units to meters
    # --------------------------------------------------
    chomp_history_m = []

    for path_xy in chomp_history:
        path_m = path_xy.copy()
        path_m[:, 0] *= ch.sx
        path_m[:, 1] *= ch.sy
        chomp_history_m.append(path_m)

    xi_full_m = xi_full.copy()
    xi_full_m[:, 0] *= ch.sx
    xi_full_m[:, 1] *= ch.sy

    # --------------------------------------------------
    # 10) Save final terrain image with initial/final path
    # --------------------------------------------------
    evaluator.plotTerrainHeightGridWithPaths(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        paths_m=[
            chomp_history_m[0],
            xi_full_m[:, 0:2]
        ],
        labels=[
            "initial path",
            "optimized CHOMP-slip path"
        ],
        save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map_with_chomp_path.png",
        show=False
    )

    # --------------------------------------------------
    # 11) Save GIF animation
    # --------------------------------------------------
    evaluator.animateChompHistoryOnTerrain(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        chomp_history_m,
        save_path="/root/ros_ws/src/tracked_robot_simulator/chomp_iterations.gif",
        interval=250,
        show=False
    )

    # --------------------------------------------------
    # 12) Optional: save individual PNG frames
    # --------------------------------------------------
    evaluator.saveChompHistoryFramesOnTerrain(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        chomp_history_m,
        output_folder="/root/ros_ws/src/tracked_robot_simulator/chomp_iteration_frames",
        frame_stride=1
    )

    print("Done.")
    print("Saved final path image to:")
    print("/root/ros_ws/src/tracked_robot_simulator/terrain_height_map_with_chomp_path.png")
    print("Saved GIF to:")
    print("/root/ros_ws/src/tracked_robot_simulator/chomp_iterations.gif")



