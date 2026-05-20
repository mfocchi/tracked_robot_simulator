import numpy as np
import matplotlib
try:
    matplotlib.use("QtAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.sparse import kron, eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path
from scipy import sparse
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 10000, suppress = True)


@dataclass
class Params:
    DOF: int
    lambda_: float
    eta: float
    MAX_ITER: int
    TOL: float
    dT: float
    t0: float
    tf: float
    convex_hull_contact: bool

@dataclass
class Robot:
    body_pts_bf: np.ndarray   # K x 2 body points in base frame
    poly_bf: np.ndarray       # N x 2 polygon in base frame
    q: np.ndarray             # [x, y, theta] pose in world


@dataclass
class MapData:
    mp: np.ndarray            # binary occupancy grid
    xRange: np.ndarray
    yRange: np.ndarray
    rows: int
    cols: int
    xL: float
    yL: float
    x: np.ndarray
    y: np.ndarray
    obstacle_cost: np.ndarray
    gradcx: np.ndarray
    gradcy: np.ndarray



class ChompSolver:
    """
    Python  implementation of a chompsolver

    Usage

        ch = ChompSolver()
        final_traj = ch.run()
    """

    def __init__(self, task_name='obs'):
        self.task_name = task_name
        pass

    # ----------------------------------------------------------------------
    # Robot / kinematics
    # ----------------------------------------------------------------------
    def createRobot(self, poly_x, poly_y, q0, map_data, convex_hull_contact):
        """
        Translation of MATLAB createRobot.
        poly_x, poly_y: polygon vertices in base frame (world units).
        q0: initial pose [x, y, theta] in world.
        """

        X = np.asarray(poly_x).reshape(-1, 1)
        Y = np.asarray(poly_y).reshape(-1, 1)
        poly_bf = np.hstack([X, Y])  # N x 2

        # Body sampling
        if not convex_hull_contact:
            # sample interior points on a grid, spacing tied to map resolution
            Dx = map_data.xL / map_data.cols
            Dy = map_data.yL / map_data.rows
            ds = 0.75 * min(Dx, Dy)
            ds = 5.0  # as in MATLAB code

            minx, maxx = poly_x.min(), poly_x.max()
            miny, maxy = poly_y.min(), poly_y.max()

            xs = np.arange(minx, maxx + ds, ds)
            ys = np.arange(miny, maxy + ds, ds)
            XX, YY = np.meshgrid(xs, ys)
            grid_points = np.vstack([XX.ravel(), YY.ravel()]).T

            path = Path(poly_bf)
            inside = path.contains_points(grid_points)
            body_pts_bf = grid_points[inside]
        else:
            # just use polygon vertices (or you could densify along edges)
            body_pts_bf = poly_bf.copy()

        # Robot pose
        robot = Robot(
            body_pts_bf=body_pts_bf,
            poly_bf=poly_bf,
            q=np.array(q0, dtype=float),
        )
        return robot

    def robotPointsWorld(self, robot, q):
        """
        Pw = robotPointsWorld(robot, q)
        q = [x, y, theta]
        """
        x, y, th = q
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        Pw = (R @ robot.body_pts_bf.T).T
        Pw[:, 0] += x
        Pw[:, 1] += y
        return Pw

    def robotPointJacobiansSE2(self, robot, q):
        """
        J = robotPointJacobiansSE2(robot, q)
        Return a list/array of Jacobians for each body point:
        J_k is 2 x 3, mapping [vx, vy, omega]^T to body point velocity.
        """
        _, _, th = q
        c = np.cos(th)
        s = np.sin(th)
        R = np.array([[c, -s],
                      [s,  c]])

        # points in world (only for cross term)
        Pw_rel = (R @ robot.body_pts_bf.T).T  # rotated (no translation)

        J_list = []
        for p in Pw_rel:
            px, py = p
            # derivative wrt [x, y, theta]
            J = np.array([
                [1.0, 0.0, -py],
                [0.0, 1.0,  px]
            ])
            J_list.append(J)
        return J_list

    # ----------------------------------------------------------------------
    # CHOMP core
    # ----------------------------------------------------------------------
    def chompSolve(self, params, robot, M, xi, h_image_metric=None):
        """
        finalTrajectory = chompSolver(params, robot, M, xi)
        """
        DOF = params.DOF
        lambda_ = params.lambda_
        MAX_ITER = params.MAX_ITER
        TOL = params.TOL

        T = xi.shape[0]

        xi_int = xi[1:T-1,:]
        N_int = T - 2 # Number of free waypoints

        # Precompute A, b, c (smoothness) The Dynamics (Hessian) Matrix A
        A, b_vec, c = self.calculate_A_matrix(T, DOF, xi)

        print("Starting CHOMP optimization...")
        # --- plotting setup ---
        h_traj = None
        title_handle = None
        if h_image_metric is not None:
            plt.figure(h_image_metric.number)
            # create a single line object for the trajectory
            (h_traj,) = plt.plot(xi[:, 0], xi[:, 1], "bo-", linewidth=2,     markersize=5,label="Current traj")
            title_handle = plt.title("Iteration 0")
            plt.legend()
            plt.draw()
            plt.pause(0.01)
        # --------------------

        for it in range(1, MAX_ITER + 1):
            print(f"\nIteration {it:3d} |", end="")

            # Step 1: total functional gradient
            (nabla_U_vec, nabla_Ftask_vec, nabla_Fsmooth_vec, total_cost,  Ftask_cost, Fsmooth_cost) \
                = self.calculate_total_gradient(xi, T, robot, params, M, A, b_vec, c, lambda_, DOF)

            # todo check  nabla_Ftask_vec is twice higher than the matlab one

            # Step 2: covariant update: delta_xi = -A^{-1} * nabla_U
            delta_xi_vec = -spsolve(A, nabla_U_vec)

            # reshape back to internal (T-2 x DOF)
            delta_xi_int = delta_xi_vec.reshape((DOF, N_int)).T  # (Nint x DOF)

            #init eta to default value
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
                    xi_try[T-1, :] = xi[T-1, :]
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
                    #print(f"\n        prev cost {total_cost:6.0f} |      new_cost {total_cost_try:6.0f}, backtracking updating  eta to {eta_try:f} \n", end="")


            # --- Step 5: Update Trajectory ---
            xi_int = xi_int_new
         
            # Reconstruct full trajectory for the next iteration (fixed boundaries)
            xi[1:T - 1, :] = xi_int

            # Logging
            grad_s_norm = np.linalg.norm(lambda_ * nabla_Fsmooth_vec)
            grad_o_norm = np.linalg.norm(nabla_Ftask_vec)
            step_norm = np.linalg.norm(eta * delta_xi_int)

            print(f"   total Cost: {total_cost:8.0f} |"
                  f"    Smoothness cost: {lambda_ * Fsmooth_cost:8.0f} |"
                  f"    Obstacle cost: {Ftask_cost:8.0f} |"
                  f"    Smooth Grad.norm: {grad_s_norm:8.0f} |"
                  f"    obs.Grad.norm: {grad_o_norm:8.0f} |"
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
        return xi_full

    def calculate_total_gradient(self, xi, T, robot, params, M, A, b_vec, c, lambda_, DOF):
        """
        [nabla_U_vec, nabla_Ftask_vec, nabla_Fsmooth_vec,
         total_cost, Fotask_cost, Fsmooth_cost]

        In this variant, theta is NOT an optimization variable:
          - xi is (T x 2) containing [x, y]
          - task term is computed using a reconstructed theta(x,y) internally
          - gradients are returned only w.r.t. [x, y] internal waypoints

        nabla vectors are stacked DOF-major: [x_block; y_block]
        """
        dT = params.dT
        Nint = T - 2

        # internal points only (XY)
        xi_int = xi[1:-1, :]               # (Nint x 2)
        xi_int_vec = xi_int.T.reshape(-1)  # DOF-major, matches A for DOF=2

        # Smoothness cost: 0.5 * x^T A x + b^T x + c
        Fsmooth_cost = 0.5 * (xi_int_vec.T @ (A @ xi_int_vec)) + np.dot(b_vec, xi_int_vec) + c
        nabla_Fsmooth_vec = (A @ xi_int_vec) + b_vec

        # Obstacle term (computed on full [x,y,theta(xy)] but returned for XY only)
        nabla_Ftask_vec, Ftask_cost = self.chompFtask_xy(xi, robot, M, dT)

        # Combine
        nabla_U_vec = lambda_ * nabla_Fsmooth_vec + nabla_Ftask_vec
        total_cost = lambda_ * Fsmooth_cost + Ftask_cost

        return (nabla_U_vec,
                nabla_Ftask_vec,
                nabla_Fsmooth_vec,
                total_cost,
                Ftask_cost,
                Fsmooth_cost)



    def chompFobs(self, xi, robot, M, dt, DOF):
        """
        [nabla_Fobs_vec, Fobs_cost] = chompFobs(xi, robot, M, dt, DOF)
        Direct translation of the provided MATLAB function.

        Inputs:
          xi: (N x 3) array of [x, y, theta]
          robot.body_pts_bf: (K x 2) body points in robot frame
          M: map object (must have obstacle_cost, gradcx, gradcy, rows, cols, xRange, yRange, ...)
          dt: timestep (float)
          DOF: degrees of freedom (int, typically 3)

        Outputs:
          nabla_Fobs_vec: 1D numpy array of length DOF*(N-2), column-major stacked (MATLAB's nabla_Fobs(:))
          Fobs_cost: scalar
        """
        import numpy as np

        eps_v = 1e-2  # same as MATLAB
        N = xi.shape[0]
        K = robot.body_pts_bf.shape[0]

        # N-2 x DOF matrix (internal timesteps only)
        nabla_Fobs = np.zeros((N - 2, DOF), dtype=float)
        Fobs_cost = 0.0

        # Precompute cos and sin for all theta
        cth = np.cos(xi[:, 2])
        sth = np.sin(xi[:, 2])

        # loop over internal timesteps (MATLAB: for t = 2:N-1)
        for t in range(1, N - 1):  # python 0-based: 1 .. N-2
            q = xi[t, :]
            q_prev = xi[t - 1, :]
            q_next = xi[t + 1, :]

            # Rotation matrices at t, t-1, t+1
            Rt = np.array([[cth[t], -sth[t]],
                           [sth[t], cth[t]]])
            Rt_prev = np.array([[cth[t - 1], -sth[t - 1]],
                                [sth[t - 1], cth[t - 1]]])
            Rt_next = np.array([[cth[t + 1], -sth[t + 1]],
                                [sth[t + 1], cth[t + 1]]])

            base = q[0:2].reshape(2, 1)  # column
            base_prev = q_prev[0:2].reshape(2, 1)
            base_next = q_next[0:2].reshape(2, 1)

            # iterate body points
            for body_point in range(K):
                r_u = robot.body_pts_bf[body_point, :].reshape(2, 1)  # column

                # body point positions
                x_u = base + Rt.dot(r_u)  # 2x1
                x_u_prev = base_prev + Rt_prev.dot(r_u)
                x_u_next = base_next + Rt_next.dot(r_u)

                # velocity and acceleration (finite differences)
                xdot = (x_u_next - x_u_prev) / (2.0 * dt)  # 2x1
                xddot = (x_u_next - 2.0 * x_u + x_u_prev) / (dt ** 2)  # 2x1

                # unit tangent (robustified)
                xdot_norm = np.linalg.norm(xdot)
                xdot_hat = (xdot / (xdot_norm + eps_v)).reshape(2, 1)

                # projector onto normal (I - v_hat v_hat^T)
                P = np.eye(2) - (xdot_hat @ xdot_hat.T)

                # centrifugal_accel = P * xddot
                centrifugal_accel = P.dot(xddot).reshape(2, 1)

                # curvature-like term kappa = centrifugal_accel / (norm(xdot)^2 + eps_v)
                denom = (xdot_norm ** 2) + eps_v
                kappa = (centrifugal_accel / denom).reshape(2, 1)  # 2x1

                # map body point position to grid coordinates (xg, yg) in continuous grid units
                xg, yg = self.worldToGridUnits(float(x_u[0, 0]), float(x_u[1, 0]), M)

                # sample obstacle cost and gradients by bilinear interpolation
                c = self.bilinearSample(M.obstacle_cost, xg, yg)  # scalar
                gcx = self.bilinearSample(M.gradcx, xg, yg)
                gcy = self.bilinearSample(M.gradcy, xg, yg)
                gradc = np.array([[gcx], [gcy]])  # 2x1 column vector

                # f = norm(xdot) * (P * gradc) - c * kappa
                f = (xdot_norm * (P.dot(gradc)) - c * kappa).reshape(2,1)  # 1D length-2 vector

                # Jacobian J = d x_u / d q   (2 x 3)
                dx = float(x_u[0] - q[0])
                dy = float(x_u[1] - q[1])
                J = np.array([[1.0, 0.0, -dy],
                              [0.0, 1.0, dx]])  # 2x3

                # map back to configuration space: J' * f  -> 3x1
                jf = J.T.dot(f.reshape(2, 1)).reshape(3, )  # 1D length-3

                # accumulate into nabla_Fobs row (t-1 corresponds to MATLAB's t-1)
                nabla_Fobs[t - 1, :] += jf

                # cost accumulation (arc-length weighted) eq 23 for each body point
                Fobs_cost += c * xdot_norm * dt

        # flatten to vector with MATLAB-style column-major ordering nabla_Fobs(:)
        nabla_Fobs_vec = nabla_Fobs.flatten(order='F')  # 1D array length of dimension DOFS(N-2)

        return nabla_Fobs_vec, Fobs_cost

    # ----------------------------------------------------------------------
    # Sampling helpers
    # ----------------------------------------------------------------------
    import numpy as np

    def bilinearSample(self, cost, xg, yg):
        """
        val = bilinearSample(cost, xg, yg)

        Exact translation of the MATLAB function.

        cost : 2D numpy array (rows x cols)
        xg   : grid x-coordinate (column index, float)
        yg   : grid y-coordinate (row index, float)
        """

        rows, cols = cost.shape

        # clamp inside valid range for bilinear (MATLAB: 1 .. cols-1 / rows-1)
        xg = max(1.0, min(cols - 1.0, xg))
        yg = max(1.0, min(rows - 1.0, yg))

        # MATLAB-style floor
        x1 = int(np.floor(xg))
        x2 = x1 + 1
        y1 = int(np.floor(yg))
        y2 = y1 + 1

        dx = xg - x1
        dy = yg - y1

        # MATLAB is 1-based, Python is 0-based → subtract 1 from indices
        c11 = cost[y1 - 1, x1 - 1]
        c12 = cost[y2 - 1, x1 - 1]
        c21 = cost[y1 - 1, x2 - 1]
        c22 = cost[y2 - 1, x2 - 1]

        val = (
                (1.0 - dx) * (1.0 - dy) * c11 +
                (1.0 - dx) * dy * c12 +
                dx * (1.0 - dy) * c21 +
                dx * dy * c22
        )

        return float(val)

    def worldToGridUnits(self, xw, yw, M):
        """
        [xg, yg] = worldToGridUnits(xw, yw, M)

        xw, yw : world coordinates (meters)
        returns:
            xg : column index in grid coordinates
            yg : row index in grid coordinates
        """

        # x and y resolutions (world units per cell)
        Dx = M.xL / M.cols
        Dy = M.yL / M.rows

        # convert world coordinates to grid units
        xg = xw / Dx
        yg = M.rows - (yw / Dy)

        return xg, yg
    

    # ----------------------------------------------------------------------
    # Theta reconstruction from XY (theta is NOT optimized)
    # ----------------------------------------------------------------------
    def addThetaFromXY(self, xi_xy, theta0=None):
        """
        Build a full SE(2) trajectory (T x 3) from an XY-only trajectory (T x 2)
        by setting theta to the local tangent direction of the curve.

        Endpoints use one-sided differences; internal points use centered differences.
        If theta0 is provided, it is used for the first waypoint only (optional).
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

        for t in range(1, T-1):
            dx = x[t+1] - x[t-1]
            dy = y[t+1] - y[t-1]
            theta[t] = np.arctan2(dy, dx)

        if theta0 is not None:
            theta[0] = float(theta0)

        xi_full = np.column_stack([x, y, theta])
        return xi_full

    def chompFtask_xy(self, xi_xy, robot, M, dt):
        """
        Wrapper: compute obstacle gradient/cost using a full (x,y,theta) trajectory where
        theta is reconstructed from XY tangent, but return gradient only w.r.t. (x,y)
        since theta is not an optimization variable.

        Returns:
          nabla_Fobs_xy_vec: length 2*(T-2), stacked like [x_block; y_block]
          Fobs_cost: scalar
        """
        xi_full = self.addThetaFromXY(xi_xy)

        func = getattr(self, "chompF" + self.task_name, None)
        if func is None:
            raise ValueError(f"Unknown task: {self.task_name}")
        nabla_full_vec, Fobs_cost = func(xi_full, robot, M, dt, DOF=3)


        T = xi_xy.shape[0]
        Nint = T - 2
        # Recover (Nint x 3) matrix from MATLAB-style column-major vec
        nabla_full = np.reshape(nabla_full_vec, (Nint, 3), order='F')  # columns: x,y,theta
        nabla_xy = nabla_full[:, 0:2]                                  # keep x,y
        nabla_xy_vec = nabla_xy.flatten(order='F')
        return nabla_xy_vec, Fobs_cost

    # ----------------------------------------------------------------------
    # Smoothness matrix A
    # ----------------------------------------------------------------------
    def calculate_A_matrix(self, T, DOF, xi0):
        """
        [A, b_vec, c] = calculate_A_matrix(T, DOF, xi0)

        Calculates the Hessian matrix A = K' * K for the smoothness cost,
        exactly translated from the MATLAB version.
        A is the Hessian of the smoothness cost (Fsmooth) in the discrete domain [9] and approximates the second derivative (acceleration) operator.
        since xi is a  uniform discretization which samples the trajectory function over equal time steps of length
        A is sparse and band diagonal, allowing O(n) solving [10].

        Inputs:
          - T: number of waypoints (int)
          - DOF: degrees of freedom (int)
          - xi0: initial trajectory, shape (T, DOF) (numpy array)

        Outputs:
          - A: sparse band-diagonal Hessian (scipy.sparse.csc_matrix) of shape (DOF*(T-2), DOF*(T-2))
          - b_vec: numpy 1-D array of length DOF*(T-2) (column-major ordering to match MATLAB's b_mat(:))
          - c: scalar float
        """
        # build full tridiagonal K_full (T x T) with [1, -2, 1] on sub, main, super diagonals
        # MATLAB: K_full = gallery('tridiag', T, 1, -2, 1);
        diagonals = [np.ones(T-1), -2*np.ones(T), np.ones(T-1)]
        offsets = [-1, 0, 1]
        K_full = sparse.diags(diagonals, offsets, shape=(T, T), format='csc')

        # select interior rows 2..T-1 (MATLAB 1-based). Python 0-based -> rows 1 .. T-2 inclusive
        K = K_full[1:(T-1), :].tocsc()   # (T-2) x T

        # Afull = K' * K  (T x T)
        Afull = (K.T).dot(K).tocsc()

        # indices (MATLAB): idx_int = 2:T-1  idx_fix = [1 T]
        # convert to 0-based indices for Python:
        idx_int = np.arange(1, T-1)        # length T-2
        idx_fix = np.array([0, T-1], dtype=int)

        # extract sub-blocks (sparse indexing returns sparse matrices)
        A_int_int = Afull[idx_int[:, None], idx_int].tocsc()   # (T-2) x (T-2)
        A_int_fix = Afull[idx_int[:, None], idx_fix].toarray() # (T-2) x 2  -> small dense (for multiplication)
        A_fix_fix = Afull[idx_fix[:, None], idx_fix].toarray() # 2 x 2 dense

        # qfix = [xi0(1,:); xi0(T,:)]  (MATLAB row 1 and row T)
        # Convert to numpy array with shape (2, DOF)
        qfix = np.vstack([xi0[0, :], xi0[-1, :]])   # (2 x DOF)

        # b_mat = A_int_fix * qfix  -> (T-2) x DOF
        # A_int_fix is (T-2 x 2), qfix is (2 x DOF)
        b_mat = A_int_fix.dot(qfix)   # dense (T-2) x DOF

        # Build A as kron(eye(DOF), A_int_int)
        A = sparse.kron(sparse.eye(DOF, format='csc'), A_int_int, format='csc')  # shape (DOF*(T-2), DOF*(T-2))

        # b_vec = b_mat(:)  MATLAB column-major stacking -> use order='F'
        b_vec = np.reshape(b_mat, (-1,), order='F')   # 1-D array length DOF*(T-2)

        # c = 0.5*sum(diag(qfix'*A_fix_fix*qfix))
        # Equivalent to 0.5 * trace(qfix.T @ A_fix_fix @ qfix)
        c = 0.5 * np.trace(qfix.T.dot(A_fix_fix).dot(qfix))

        # debug
        # A_dense = A.toarray()
        # print(A_dense)
        # plt.spy(A, markersize=1)
        # plt.title("Sparsity pattern of A")
        # plt.show()


        return A, b_vec, c
    # ----------------------------------------------------------------------
    # Map construction
    # ----------------------------------------------------------------------
    def convertToMap(self, X, Y, map_data):
        """
        [Xp, Yp] = convertToMap(X, Y, map)
        Convert polygon vertices from world coordinates to pixel indices.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        x_min, x_max = map_data.xRange
        y_min, y_max = map_data.yRange

        u = (X - x_min) / (x_max - x_min)
        v = (Y - y_min) / (y_max - y_min)

        Xp = u * (map_data.cols - 1)
        Yp = (1.0 - v) * (map_data.rows - 1)

        return Xp, Yp

    def constructMap(self, xRange, yRange, rows, cols, obstacles, epsilon):
        """
        M = constructMap(xRange, yRange, rows, cols, obstacles, epsilon)
        Rough translation of the MATLAB map construction:
        - binary occupancy from polygon obstacles
        - signed distance field
        - obstacle cost
        - gradient of cost
        """
        mp = np.zeros((rows, cols), dtype=bool)

        # rasterize each obstacle polygon
        Y_indices, X_indices = np.indices((rows, cols))
        points = np.vstack([X_indices.ravel(), Y_indices.ravel()]).T

        for obs in obstacles:
            Xw = obs["X"]
            Yw = obs["Y"]
            Xp, Yp = self.convertToMap(Xw, Yw,  # map placeholder
                                       type("Tmp", (), {
                                           "xRange": xRange,
                                           "yRange": yRange,
                                           "rows": rows,
                                           "cols": cols
                                       })())
            poly = np.vstack([Xp, Yp]).T
            path = Path(poly)
            inside = path.contains_points(points)
            inside_mask = inside.reshape((rows, cols))
            mp |= inside_mask

        # Euclidean distance transforms
        D = distance_transform_edt(~mp)   # distance to obstacle (outside)
        D1 = distance_transform_edt(mp)   # distance inside obstacle

        # signed distance field: positive outside, negative inside
        H = D - D1

        # obstacle cost field
        eps = epsilon
        obstacle_cost = np.zeros_like(H, dtype=float)

        mask1 = H < 0
        obstacle_cost[mask1] = -H[mask1] + eps / 2.0

        mask2 = (H > 0) & (H <= eps)
        obstacle_cost[mask2] = (1.0 / (2.0 * eps)) * (H[mask2] - eps)**2

        # coordinate arrays in grid units
        Mx = (np.arange(cols) + 0.5)
        My = (np.arange(rows) + 0.5)

        xL = xRange[1] - xRange[0]
        yL = yRange[1] - yRange[0]

        x_resolution = xL / cols
        y_resolution = yL / rows

        # convert to world units
        X_world = x_resolution * Mx
        Y_world = y_resolution * (rows - My)  # flip vertically

        # gradient spacing
        Dx = xL / cols
        Dy = yL / rows

        gradcy, gradcx = np.gradient(obstacle_cost, Dy, Dx)
        # y increases downward in image, so flip sign
        gradcy = -gradcy

        M = MapData(
            mp=mp,
            xRange=xRange.astype(float),
            yRange=yRange.astype(float),
            rows=rows,
            cols=cols,
            xL=xL,
            yL=yL,
            x=X_world,
            y=Y_world,
            obstacle_cost=obstacle_cost,
            gradcx=gradcx,
            gradcy=gradcy,
        )
        # # debug cost
        # plt.figure()
        # plt.spy(M.obstacle_cost, markersize=1)
        # plt.title("Sparsity pattern of Obstacle cost")
        # plt.show()
        #
        # #debug gradient
        # Dx = M.xL / M.cols
        # Dy = M.yL / M.rows
        # x = Dx * (np.arange(M.cols) + 0.5)
        # y = Dy * (np.arange(M.rows) + 0.5)
        # Xw, Yw = np.meshgrid(x, y)
        #
        # plt.figure()
        # plt.imshow(np.flipud(M.obstacle_cost), extent=[x.min(), x.max(), y.min(), y.max()],
        #            origin="lower")
        # plt.title("Python world coords")
        # plt.colorbar()
        #
        # Gx = np.flipud(M.gradcx)
        # Gy = np.flipud(M.gradcy)
        #
        # step = 20
        # plt.quiver(Xw[::step, ::step], Yw[::step, ::step],
        #            Gx[::step, ::step], Gy[::step, ::step],
        #            angles="xy", scale_units="xy", scale=0.8)
        # plt.gca().set_aspect("equal")
        # plt.show()
        return M

    # ----------------------------------------------------------------------
    # Plotting helper (map)
    # ----------------------------------------------------------------------
    def plotMap(self, M):
        """
        h = plotMap(M)
        Show obstacle map and cost field in world coordinates.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(M.xRange[0], M.xRange[1])
        ax.set_ylim(M.yRange[0], M.yRange[1])
        ax.set_aspect("equal")

        # plot obstacle cost as an image in world coordinates
        extent = [M.xRange[0], M.xRange[1], M.yRange[0], M.yRange[1]]
        im = ax.imshow(
            np.flipud(M.obstacle_cost),
            extent=extent,
            origin="lower",
            interpolation="nearest",
            alpha=0.7,
        )
        plt.colorbar(im, ax=ax, label="Obstacle cost")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Map and obstacle cost field")
        return fig

    def obstacles_to_stl_scaled(self, obstacles, out_stl_path, height_m=1.0,  sx=1.0, sy=1.0,    origin_xy_m=(0.0, 0.0) ):
        """
        obstacles: list of {"X": [...], "Y": [...]} in NON-metric units
        sx, sy: meters per unit in x,y
        origin_xy_m: (ox, oy) added after scaling, in meters (optional)
        """
        import trimesh
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        ox, oy = origin_xy_m

        polys = []
        for obs in obstacles:
            X = np.asarray(obs["X"], dtype=float).ravel()
            Y = np.asarray(obs["Y"], dtype=float).ravel()

            # scale to meters + optional shift
            Xm = sx * X + ox
            Ym = sy * Y + oy

            # drop repeated last point if present
            if len(Xm) > 2 and np.isclose(Xm[0], Xm[-1]) and np.isclose(Ym[0], Ym[-1]):
                Xm = Xm[:-1]
                Ym = Ym[:-1]

            if len(Xm) < 3:
                continue

            poly = Polygon(np.column_stack([Xm, Ym]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area <= 1e-12:
                continue

            polys.append(poly)

        if not polys:
            raise ValueError("No valid obstacle polygons to export.")

        merged = unary_union(polys)
        geoms = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)

        meshes = [trimesh.creation.extrude_polygon(g, height=height_m) for g in geoms]
        mesh = trimesh.util.concatenate(meshes)

        mesh.remove_unreferenced_vertices()
        mesh.process(validate=True)

        mesh.export(out_stl_path)
        return out_stl_path

def optimize(self, xi0, M, params, robot, return_history=False, save_every=1, callback=None):
    # plot
    plt.ion()
    h_image_metric = self.plotMap(M)
    plt.figure(h_image_metric.number)
    # plot initial trajectory
    plt.plot(xi0[:, 0], xi0[:, 1], "ko-", linewidth=2, label="Initial")
    # optimize
    optimized_xi = self.chompSolve(params, robot, M, xi0, h_image_metric)
    plt.figure(h_image_metric.number)
    # plot optimized trajectory
    plt.plot(optimized_xi[:, 0], optimized_xi[:, 1], "ro-", linewidth=2, label="CHOMP")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # plot orientation (debug)
    # compute tangent angle
    # N = optimized_xi.shape[0]
    # theta_des = np.zeros(N)
    # xs = optimized_xi[:, 0]
    # ys = optimized_xi[:, 1]
    # for t in range(1, optimized_xi.shape[0] - 1):
    #     dx = xs[t + 1] - xs[t - 1]
    #     dy = ys[t + 1] - ys[t - 1]
    #     theta_des[t] = math.atan2(dy, dx)
    # # copy into initial and final sample the neightbours
    # theta_des[0] = theta_des[1]
    # theta_des[N - 1] = theta_des[N - 2]
    #
    # plt.figure()
    # plt.plot(theta_des, "bo-", linewidth=2, markersize=2, label="TANGENT")
    # plt.plot(optimized_xi[:, 2], "ro-", linewidth=1, markersize=1, label="CHOMP")
    # plt.grid(True)
    # plt.title("Theta plot")
    # plt.ylabel('theta [rad]')
    # plt.axis("equal")
    # plt.show()

    return optimized_xi


# If you want to run it directly:
if __name__ == "__main__":

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

    # --------------------------------
    # 2) Robot + CHOMP parameters
    # --------------------------------
    theta0 = np.pi / 4.0

    # Optimize ONLY [x, y]. Theta is reconstructed from the XY tangent.
    q_start = np.array([0.0, 0.0, theta0])
    q_goal  = np.array([300.0, 200.0, theta0])
    # q_goal  = np.array([450.0, 400.0, theta0])
    # q_goal  = np.array([400.0, 100.0, theta0])


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
    v = np.hypot(dx, dy)/ chomp_params.dT
    omega = dtheta/ chomp_params.dT

    # append last value to keep same length N of optimized_xi_meters
    dx = np.append(dx, dx[-1])
    dy = np.append(dy, dy[-1])
    dtheta = np.append(dtheta, dtheta[-1])

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

