import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from chomp_no_theta import Params, ChompSolver
from base_controllers.evaluate_energy_consumption  import EvaluateEnergyConsumption
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

    def energy_cost(self, xi_xy, dt):
        """
        Black-box energy functional:
            F_energy(xi) = computeCost(...)
        """
        des_x, des_y, des_theta, v_des, omega_des = self.trajectory_to_motion(xi_xy, dt)

        cost = self.evaluateEnergyConsumption.computeCost(
            des_x_vec=des_x,
            des_y_vec=des_y,
            des_yaw_vec=des_theta,
            v_ol=v_des,
            omega_ol=omega_des,
            plan_dt=dt
        )
        return cost

    # ------------------------------------------------------------------
    # Energy functional gradient by finite differences
    # ------------------------------------------------------------------
    def chompFslip_old(self, xi_xy, robot, M, dt, DOF):
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

    def chompFslip(self, xi_xy, robot, M, dt, DOF):
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
        x_plus[:,0] += self.fd_eps
        x_minus[:,0] -= self.fd_eps
        cost_plus_x = self.energy_cost(x_plus, dt)
        cost_minus_x = self.energy_cost(x_minus, dt)
        grad_x = (cost_plus_x - cost_minus_x) / (2.0 * self.fd_eps)
        #grad along Y
        y_plus = xi_xy.copy()
        y_minus = xi_xy.copy()
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
        return nabla_Fenergy_vec, Fenergy_cost

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
        nabla_Ftask_vec, Ftask_cost = self.chompFslip(xi, robot, M, dT, DOF)

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

    # ------------------------------------------------------------------
    # Optional: custom optimize wrapper for plotting labels
    # ------------------------------------------------------------------
    def optimize(self, xi0, M, params, robot):
        plt.ion()
        h_image_metric = self.plotMap(M)
        plt.figure(h_image_metric.number)
        plt.plot(xi0[:, 0], xi0[:, 1], "ko-", linewidth=2, label="Initial")

        optimized_xi = self.chompSolve(params, robot, M, xi0, h_image_metric)

        plt.figure(h_image_metric.number)
        xi_full = self.addThetaFromXY(optimized_xi)
        plt.plot(xi_full[:, 0], xi_full[:, 1], "ro-", linewidth=2, label="CHOMP-energy")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        return xi_full

# If you want to run it directly:
if __name__ == "__main__":


    # initial pose
    p0 = np.array([0., 0., 0.])
    # final pose
    pf = np.array([220 * 0.02, 190 * 0.02, np.pi / 4])  # 0.02 is the conversion gain to convert units used in chomp_no_theta into meters

    ch = ChompSolverSlip()
    # -------------------------------
    # 1) Create a map (as in script)
    # -------------------------------
    # obstacles: list of dicts with X, Y in world coordinates
    obstacles = [{"X": np.array([150, 350, 350, 150]),
         "Y": np.array([50, 50, 150, 150])},
        {"X": np.array([200, 300, 250]),
         "Y": np.array([300, 300, 400])}, ]

    #map origin
    xRange = np.array([0.0, 500.0])
    yRange = np.array([0.0, 500.0])

    rows = 2000
    cols = 2000
    epsilon = 50.0

    M = ch.constructMap(xRange, yRange, rows, cols, obstacles, epsilon)

    # create metric stl for rviz
    # your current world extents (in "world units")
    xL_world = xRange[1] - xRange[0]
    yL_world = yRange[1] - yRange[0]

    # desired real size in meters
    xL_m_des = 10.0  # e.g. want the map width to be 50m
    yL_m_des = 10.0

    #meter to world_unit
    ch.sx = xL_m_des / xL_world
    ch.sy = yL_m_des / yL_world
    import rospkg
    ch.obstacles_to_stl_scaled(obstacles, rospkg.RosPack().get_path('tractor_description') + '/meshes/obstacles.stl', height_m=2.0, sx=ch.sx, sy=ch.sy)

    chomp_params = Params(
        DOF=2,
        lambda_=200.0,
        eta=0.001,
        MAX_ITER=100,
        TOL=1.0,
        dT=1.0,
        t0=0.0,
        tf=20.0,
        convex_hull_contact=True,
    )

    # map from metric to world units (expt for theta)
    q_start = p0.copy()
    q_start[:2] /=ch.sx
    q_goal = pf.copy()
    q_goal[:2] /= ch.sy

    # polygon in base frame (world units)
    #robot size
    w = 60.0
    h = 40.0
    X = np.array([-w / 2, w / 2, w / 2, -w / 2], dtype=float)
    Y = np.array([-h / 2, -h / 2, h / 2, h / 2], dtype=float)
    robot = ch.createRobot(X, Y, q_start, M, chomp_params.convex_hull_contact)

    # --------------------------------
    # 3) Initial straight-line trajectory
    # --------------------------------
    T = int(chomp_params.tf / chomp_params.dT)
    xi0 = np.zeros((T, 2), dtype=float)
    xi0[:, 0] = np.linspace(q_start[0], q_goal[0], T)
    xi0[:, 1] = np.linspace(q_start[1], q_goal[1], T)

    optimized_xi = ch.optimize(xi0, M, chomp_params, robot)

    #map to meters
    optimized_xi_meters = optimized_xi.copy()
    optimized_xi_meters[:, 0] *= ch.sx
    optimized_xi_meters[:, 1] *= ch.sy

    #compute velocities
    dx = np.diff(optimized_xi_meters[:, 0])
    dy = np.diff(optimized_xi_meters[:, 1])
    dtheta = np.diff(np.unwrap(optimized_xi_meters[:, 2]))

    # append last value to keep same length N of optimized_xi_meters
    dx = np.append(dx, dx[-1])
    dy = np.append(dy, dy[-1])
    dtheta = np.append(dtheta, dtheta[-1])

    v = np.hypot(dx, dy)/ chomp_params.dT
    omega = dtheta/ chomp_params.dT


    plt.figure()
    plt.plot(v, "bo-", linewidth=2, markersize=2, label="long")
    plt.ylim([-1,1])
    plt.grid(True)
    plt.ylabel("v")

    plt.figure()
    plt.plot(omega, "ro-", linewidth=1, markersize=1, label="omega")
    plt.ylabel("omega")
    plt.ylim([-1, 1])
    plt.grid(True)
    plt.axis("equal")
    plt.show()




