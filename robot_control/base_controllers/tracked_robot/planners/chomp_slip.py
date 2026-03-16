import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from chomp_no_theta import Params, ChompSolver
from base_controllers.evaluate_energy_consumption  import initializeEnergyComputation, computeCost

class ChompSolverSlip(ChompSolver):
    def __init__(self):
        pass

    def chompFSlip(self, xi, computeCost, dt, DOF):
        """
        Inputs:
          xi: (N x 3) array of [x, y, theta]

          dt: timestep (float)
          DOF: degrees of freedom (int, typically 3)

        Outputs:
          nabla_Fslip_vec: 1D numpy array of length DOF*(N-2), column-major stacked (MATLAB's nabla_Fobs(:))
          Fslip_cost: scalar
        """
        #initialize arrays
        N = xi.shape[0]
        # N-2 x DOF matrix (internal timesteps only)
        nabla_Fslip = np.zeros((N - 2, DOF), dtype=float)
        Fslip_cost = 0.0

        # map to meters
        xi_meters = xi.copy()
        xi_meters[:, 0] *= self.sx
        xi_meters[:, 1] *= self.sy
        # compute velocities
        dx = np.diff(xi_meters[:, 0])
        dy = np.diff(xi_meters[:, 1])
        dtheta = np.diff(np.unwrap(xi_meters[:, 2]))

        # append last value to keep same length N of optimized_xi_meters
        dx = np.append(dx, dx[-1])
        dy = np.append(dy, dy[-1])
        dtheta = np.append(dtheta, dtheta[-1])
        v_des = np.hypot(dx, dy) / params.dT
        omega_des = dtheta / params.dT

        Fslip_cost = computeCost(xi_meters, )


        #### TODO gradient computation
        # flatten to vector with MATLAB-style column-major ordering nabla_Fobs(:)
        nabla_Fslip_vec = nabla_Fslip.flatten(order='F')  # 1D array length of dimension 3(N-2)

        return nabla_Fslip_vec, Fslip_cost

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
    sx = xL_m_des / xL_world
    sy = yL_m_des / yL_world
    import rospkg
    ch.obstacles_to_stl_scaled(obstacles, rospkg.RosPack().get_path('tractor_description') + '/meshes/obstacles.stl', height_m=2.0, sx=sx, sy=sy)

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
    optimized_xi_meters[:, 0] *= sx
    optimized_xi_meters[:, 1] *= sy

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

