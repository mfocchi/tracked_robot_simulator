#!/usr/bin/env python3

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

# Adjust these imports to your real project structure
from base_controllers.tracked_robot.planners.chomp_slip import ChompSolverSlip, Params
from base_controllers.evaluate_energy_consumption import EvaluateEnergyConsumption


class ChompSolverCheapSlipRisk(ChompSolverSlip):
    """
    Experiment 5:
    Cheap terrain-based slip-risk CHOMP.

    This replaces the expensive simulated slip/energy gradient with a cheap
    cost computed from the terrain height grid.

    The cost considers:
        - forward slope risk, which further considers:
                - uphill weight
                - downhill weight
        - lateral slope risk (indifferent between uphill and downhill).
        - terrain roughness
        - path curvature
        - speed amplification of slip risk

    It is not the true simulated energy/slip cost.
    It is a fast approximation used for planning.
    """

    def __init__(self):
        super().__init__()

        # ==================================================
        # 1) Task identity
        # ==================================================
        self.task_name = "cheap slip-risk"

        # ==================================================
        # 2) Cost weights
        # ==================================================
        # Slope-related weights
        self.w_forward_slope = 1.0
        self.w_lateral_slope = 5.0

        # Directional slope / gravity-inspired weights
        # uphill_slope   = max(0, forward_slope)
        # downhill_slope = max(0, -forward_slope)
        self.w_uphill = 2.0
        self.w_downhill = 1.0

        # Terrain geometry weights
        self.w_height = 2.0
        self.w_roughness = 1.0

        # Motion-related weights
        self.w_curvature = 0.5
        self.w_speed = 1.0

        # Path length penalty
        self.w_path_length = 5.0

        # ==================================================
        # 3) Reference values for normalization
        # ==================================================
        # These define what is considered "large" or "important".
        self.forward_slope_ref = np.tan(np.deg2rad(15.0))
        self.lateral_slope_ref = np.tan(np.deg2rad(8.0))

        self.height_ref = 2.0
        self.roughness_ref = 0.05

        # curvature = 1 / turning_radius.
        # curvature_ref = 1.0 means radius around 1 meter.
        self.curvature_ref = 1.0

        # Reference speed in m/s.
        self.speed_ref = 0.5

        # ==================================================
        # 4) Safety thresholds
        # ==================================================
        # Terrain above this height starts being penalized.
        self.height_safe = 0.5

        # ==================================================
        # 5) Terrain grid/interpolator state
        # ==================================================
        self.terrain_ready = False

        self.x_grid = None
        self.y_grid = None

        self.height_interp = None
        self.slope_x_interp = None
        self.slope_y_interp = None
        self.roughness_interp = None
    # ------------------------------------------------------
    # Terrain preprocessing
    # ------------------------------------------------------
    def fill_nan_nearest(self, Z):
        """
        Fill NaN cells in the terrain height grid using nearest valid cells.
        """

        Z = np.asarray(Z, dtype=float)

        valid = np.isfinite(Z)

        if not np.any(valid):
            raise ValueError("Terrain height grid contains only NaN values.")

        if np.all(valid):
            return Z.copy()

        nan_mask = ~valid

        nearest_indices = distance_transform_edt(
            nan_mask,
            return_distances=False,
            return_indices=True
        )

        Z_filled = Z[tuple(nearest_indices)]

        return Z_filled

    def set_terrain_height_grid(self, X, Y, Z):
        """
        Precompute terrain slope and roughness maps from the terrain height grid.

        X, Y, Z come from:

            evaluator.computeTerrainHeightGrid(...)

        X, Y, Z are in meters.
        """

        Z_filled = self.fill_nan_nearest(Z)

        # Grid coordinates
        x_grid = X[0, :]
        y_grid = Y[:, 0]

        self.x_grid = x_grid
        self.y_grid = y_grid

        # --------------------------------------------------
        # First derivatives: terrain slope
        # --------------------------------------------------
        dz_dy, dz_dx = np.gradient(
            Z_filled,
            y_grid,
            x_grid,
            edge_order=2
        )

        # --------------------------------------------------
        # Second derivatives: roughness approximation
        # --------------------------------------------------
        d2z_dy2, d2z_dydx = np.gradient(
            dz_dy,
            y_grid,
            x_grid,
            edge_order=2
        )

        d2z_dxdy, d2z_dx2 = np.gradient(
            dz_dx,
            y_grid,
            x_grid,
            edge_order=2
        )

        roughness = np.sqrt(
            d2z_dx2**2
            + d2z_dy2**2
            + 2.0 * d2z_dxdy**2
        )

        # --------------------------------------------------
        # Interpolators
        # RegularGridInterpolator expects points as (y, x)
        # --------------------------------------------------
        self.slope_x_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            dz_dx,
            bounds_error=False,
            fill_value=None
        )

        self.slope_y_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            dz_dy,
            bounds_error=False,
            fill_value=None
        )

        self.roughness_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            roughness,
            bounds_error=False,
            fill_value=None
        )

        self.height_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            Z_filled,
            bounds_error=False,
            fill_value=None
        )
        self.terrain_ready = True

        print("Cheap slip-risk terrain maps created.")
        print("x range:", x_grid[0], "to", x_grid[-1])
        print("y range:", y_grid[0], "to", y_grid[-1])

    # ------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------

    def set_world_meter_transform(self, x_min_m, x_max_m, y_min_m, y_max_m, xRange, yRange):
        """
        Define affine conversion between CHOMP/world units and real terrain meters.

        CHOMP/world:
            xRange = [0, 500]
            yRange = [0, 500]

        Terrain meters:
            x in [x_min_m, x_max_m]
            y in [y_min_m, y_max_m]
        """

        self.xRange = xRange
        self.yRange = yRange

        self.x_origin_m = x_min_m
        self.y_origin_m = y_min_m

        self.sx = (x_max_m - x_min_m) / (xRange[1] - xRange[0])
        self.sy = (y_max_m - y_min_m) / (yRange[1] - yRange[0])

    def meters_to_world_xy(self, path_m):
        """
        Convert real terrain meter coordinates to CHOMP/world units.
        """

        path_m = np.asarray(path_m, dtype=float)

        path_w = path_m.copy()
        path_w[..., 0] = (path_m[..., 0] - self.x_origin_m) / self.sx
        path_w[..., 1] = (path_m[..., 1] - self.y_origin_m) / self.sy

        return path_w

    def world_to_meters_xy(self, path_w):
        """
        Convert CHOMP/world units to real terrain meter coordinates.
        """

        path_w = np.asarray(path_w, dtype=float)

        path_m = path_w.copy()
        path_m[..., 0] = self.x_origin_m + path_w[..., 0] * self.sx
        path_m[..., 1] = self.y_origin_m + path_w[..., 1] * self.sy

        return path_m

    def compute_path_theta(self, path_m):
        """
        Compute heading angle from path tangent.
        """

        dx = np.gradient(path_m[:, 0])
        dy = np.gradient(path_m[:, 1])

        theta = np.arctan2(dy, dx)

        return theta

    def compute_path_speed(self, path_m, dt):
        """
        Compute approximate path speed in m/s.
        """

        dx = np.gradient(path_m[:, 0], dt)
        dy = np.gradient(path_m[:, 1], dt)

        v = np.sqrt(dx**2 + dy**2)

        return v

    def compute_path_curvature(self, path_m, dt):
        """
        Compute approximate planar curvature of the path.

        kappa = (x_dot y_ddot - y_dot x_ddot) /
                (x_dot^2 + y_dot^2)^(3/2)
        """

        x = path_m[:, 0]
        y = path_m[:, 1]

        x_dot = np.gradient(x, dt)
        y_dot = np.gradient(y, dt)

        x_ddot = np.gradient(x_dot, dt)
        y_ddot = np.gradient(y_dot, dt)

        eps = 1e-8

        denominator = (x_dot**2 + y_dot**2 + eps) ** 1.5

        kappa = (x_dot * y_ddot - y_dot * x_ddot) / denominator

        return kappa

    # ------------------------------------------------------
    # Cheap slip-risk cost
    # ------------------------------------------------------
    def cheap_slip_risk_cost(self, xi_xy, dt):
        """
        Cheap terrain-based slip-risk cost.

        The trajectory xi_xy is in CHOMP/world units.
        Internally we convert it to meters.

        Cost:

            J = sum [
                speed_factor *
                (
                    w_f * normalized_forward_slope^2
                  + w_l * normalized_lateral_slope^2
                  + w_r * normalized_roughness^2
                )
              + w_k * normalized_curvature^2
            ] dt
        """

        if not self.terrain_ready:
            raise RuntimeError(
                "Terrain grid not set. Call set_terrain_height_grid(X, Y, Z) first."
            )

        path_m = self.world_to_meters_xy(xi_xy)

        x = path_m[:, 0]
        y = path_m[:, 1]

        points_yx = np.column_stack((y, x))

        slope_x = self.slope_x_interp(points_yx)
        slope_y = self.slope_y_interp(points_yx)
        roughness = self.roughness_interp(points_yx)

        slope_x = np.nan_to_num(slope_x, nan=0.0, posinf=0.0, neginf=0.0)
        slope_y = np.nan_to_num(slope_y, nan=0.0, posinf=0.0, neginf=0.0)
        roughness = np.nan_to_num(roughness, nan=0.0, posinf=0.0, neginf=0.0)

        theta = self.compute_path_theta(path_m)

        forward_x = np.cos(theta)
        forward_y = np.sin(theta)

        lateral_x = -np.sin(theta)
        lateral_y = np.cos(theta)

        # Slope along robot forward direction
        forward_slope = slope_x * forward_x + slope_y * forward_y

        # Slope along robot lateral direction
        lateral_slope = slope_x * lateral_x + slope_y * lateral_y

        curvature = self.compute_path_curvature(path_m, dt)
        speed = self.compute_path_speed(path_m, dt)

        # --------------------------------------------------
        # Normalized terms
        # --------------------------------------------------
        height = self.height_interp(points_yx)
        height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------
        # Directional forward slope
        # ------------------------------
        uphill_slope = np.maximum(0.0, forward_slope)
        downhill_slope = np.maximum(0.0, -forward_slope)

        uphill_term = (uphill_slope / self.forward_slope_ref) ** 2
        downhill_term = (downhill_slope / self.forward_slope_ref) ** 2

        # ------------------------------
        # Lateral slope
        # ------------------------------
        lateral_term = (lateral_slope / self.lateral_slope_ref) ** 2

        # ------------------------------
        # Roughness
        # ------------------------------
        roughness_term = (roughness / self.roughness_ref) ** 2

        # ------------------------------
        # Curvature
        # ------------------------------
        curvature_term = (curvature / self.curvature_ref) ** 2

        # ------------------------------
        # Height penalty
        # ------------------------------
        height_excess = np.maximum(0.0, height - self.height_safe)
        height_term = (height_excess / self.height_ref) ** 2

        # ------------------------------
        # Speed amplification
        # ------------------------------
        speed_term = (speed / self.speed_ref) ** 2
        speed_factor = 1.0 + self.w_speed * speed_term

        terrain_risk = (
                self.w_uphill * uphill_term
                + self.w_downhill * downhill_term
                + self.w_lateral_slope * lateral_term
                + self.w_roughness * roughness_term
                + self.w_height * height_term
        )

        turning_risk = self.w_curvature * curvature_term

        cost_density = speed_factor * terrain_risk + turning_risk

        segment_lengths = np.hypot(
            np.diff(path_m[:, 0]),
            np.diff(path_m[:, 1])
        )

        path_length = np.sum(segment_lengths)

        straight_length = np.linalg.norm(path_m[-1, :] - path_m[0, :])
        straight_length = max(straight_length, 1e-6)

        length_term = self.w_path_length * (path_length / straight_length) ** 2

        total_cost = np.sum(cost_density) * dt + length_term

        return float(total_cost)

    # ------------------------------------------------------
    # Make cheap cost compatible with existing CHOMP code
    # ------------------------------------------------------
    def energy_cost(self, xi_xy, dt):
        """
        Override the original expensive energy_cost.

        For experiment 5, energy_cost means cheap slip-risk cost.
        This allows existing CHOMP code and backtracking to work unchanged.
        """

        return self.cheap_slip_risk_cost(xi_xy, dt)

    def compute_slip_cost_gradient(self, xi_xy, robot, M, dt, DOF):
        """
        Cheap finite-difference gradient of the cheap slip-risk cost.

        This still uses finite differences over waypoints, but every cost
        evaluation is cheap because it uses terrain-grid interpolation instead
        of full robot simulation.
        """

        T = xi_xy.shape[0]
        Nint = T - 2

        if DOF != 2:
            raise ValueError(f"ChompSolverCheapSlipRisk assumes DOF=2, got DOF={DOF}")

        grad = np.zeros((Nint, DOF), dtype=float)

        Ftask_cost = self.cheap_slip_risk_cost(xi_xy, dt)

        for t in range(1, T - 1):
            for d in range(DOF):
                xi_plus = xi_xy.copy()
                xi_minus = xi_xy.copy()

                xi_plus[t, d] += self.fd_eps
                xi_minus[t, d] -= self.fd_eps

                cost_plus = self.cheap_slip_risk_cost(xi_plus, dt)
                cost_minus = self.cheap_slip_risk_cost(xi_minus, dt)

                grad[t - 1, d] = (cost_plus - cost_minus) / (2.0 * self.fd_eps)

        grad = np.clip(grad, -self.grad_clip, self.grad_clip)

        nabla_Ftask_vec = grad.flatten(order="F")

        return nabla_Ftask_vec, Ftask_cost


if __name__ == "__main__":

    # --------------------------------------------------
    # 1) Output root folder
    # --------------------------------------------------
    experiment_name = "5_cheap_slip_risk_chomp"

    output_root = (
        "/root/ros_ws/src/tracked_robot_simulator/"
        "robot_control/base_controllers/"
        "chomp_slip_workspace/outputs/"
        + experiment_name
    )

    # Clear old experiment outputs
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    # Recreate clean output folder
    os.makedirs(output_root, exist_ok=True)
    # --------------------------------------------------
    # 2) Test cases in real terrain coordinates [meters]
    # --------------------------------------------------
    test_cases = [
        (
            "case_1_high_to_low",
            np.array([0.0, 0.0]),
            np.array([10.0, -2.0])
        ),
        (
            "case_2_low_to_high",
            np.array([10.0, -2.0]),
            np.array([0.0, 0.0])
        ),
        (
            "case_3_hill_avoidance",
            np.array([-15.0, -12.0]),
            np.array([-10.0, 5.0])
        ),
    ]

    # --------------------------------------------------
    # 3) Create terrain evaluator and height grid
    # --------------------------------------------------
    evaluator = EvaluateEnergyConsumption()

    X, Y, Z, x_edges, y_edges = evaluator.computeTerrainHeightGrid(
        nx=150,
        ny=150,
        samples_per_cell=1,
        z_margin=5.0
    )

    terrain_map_path = os.path.join(
        output_root,
        "terrain_height_map.png"
    )

    evaluator.plotTerrainHeightGrid(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        save_path=terrain_map_path,
        show=False
    )

    evaluator.printTerrainHeightCandidates(
        X,
        Y,
        Z,
        n=10
    )

    # --------------------------------------------------
    # 4) CHOMP map coordinates
    # --------------------------------------------------
    # These are internal CHOMP/world units, not pixels.
    # They are only used by the CHOMP solver.
    xRange = np.array([0.0, 500.0])
    yRange = np.array([0.0, 500.0])

    rows = 2000
    cols = 2000
    epsilon = 50.0

    obstacles = []

    # Terrain bounds in meters
    x_min_m = x_edges[0]
    x_max_m = x_edges[-1]
    y_min_m = y_edges[0]
    y_max_m = y_edges[-1]

    print("\nTerrain bounds in meters:")
    print("x:", x_min_m, "to", x_max_m)
    print("y:", y_min_m, "to", y_max_m)

    # --------------------------------------------------
    # 5) CHOMP parameters
    # --------------------------------------------------
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

    T = int(chomp_params.tf / chomp_params.dT)

    # Robot footprint in meters
    robot_w_m = 1.2
    robot_h_m = 0.8

    # --------------------------------------------------
    # 6) Run all test cases
    # --------------------------------------------------
    for case_name, start_xy_m, goal_xy_m in test_cases:

        print("\n" + "=" * 80)
        print("Running:", case_name)
        print("start [m]:", start_xy_m)
        print("goal  [m]:", goal_xy_m)
        print("=" * 80)

        case_output_dir = os.path.join(
            output_root,
            case_name
        )

        os.makedirs(case_output_dir, exist_ok=True)

        # --------------------------------------------------
        # Create a fresh solver for this case
        # --------------------------------------------------
        ch = ChompSolverCheapSlipRisk()

        ch.set_world_meter_transform(
            x_min_m=x_min_m,
            x_max_m=x_max_m,
            y_min_m=y_min_m,
            y_max_m=y_max_m,
            xRange=xRange,
            yRange=yRange
        )

        print("World-to-meter scale:")
        print("sx =", ch.sx, "m/world-unit")
        print("sy =", ch.sy, "m/world-unit")

        ch.set_terrain_height_grid(
            X,
            Y,
            Z
        )

        M = ch.constructMap(
            xRange,
            yRange,
            rows,
            cols,
            obstacles,
            epsilon
        )

        # --------------------------------------------------
        # Convert start/goal from meters to CHOMP/world units
        # --------------------------------------------------
        start_goal_m = np.vstack([
            start_xy_m,
            goal_xy_m
        ])

        start_goal_w = ch.meters_to_world_xy(
            start_goal_m
        )

        start_xy_w = start_goal_w[0]
        goal_xy_w = start_goal_w[1]

        theta0 = np.arctan2(
            goal_xy_m[1] - start_xy_m[1],
            goal_xy_m[0] - start_xy_m[0]
        )

        q_start = np.array([
            start_xy_w[0],
            start_xy_w[1],
            theta0
        ])

        q_goal = np.array([
            goal_xy_w[0],
            goal_xy_w[1],
            theta0
        ])

        print("start [world units]:", q_start)
        print("goal  [world units]:", q_goal)

        # --------------------------------------------------
        # Robot footprint in CHOMP/world units
        # --------------------------------------------------
        robot_w_world = robot_w_m / ch.sx
        robot_h_world = robot_h_m / ch.sy

        X_robot = np.array(
            [
                -robot_w_world / 2.0,
                 robot_w_world / 2.0,
                 robot_w_world / 2.0,
                -robot_w_world / 2.0
            ],
            dtype=float
        )

        Y_robot = np.array(
            [
                -robot_h_world / 2.0,
                -robot_h_world / 2.0,
                 robot_h_world / 2.0,
                 robot_h_world / 2.0
            ],
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
        # Initial straight-line trajectory in CHOMP/world units
        # --------------------------------------------------
        xi0 = np.zeros((T, 2), dtype=float)

        xi0[:, 0] = np.linspace(
            q_start[0],
            q_goal[0],
            T
        )

        xi0[:, 1] = np.linspace(
            q_start[1],
            q_goal[1],
            T
        )

        # --------------------------------------------------
        # Run cheap slip-risk CHOMP
        # --------------------------------------------------
        xi_full, chomp_history = ch.optimize_with_history(
            xi0,
            M,
            chomp_params,
            robot
        )

        print("\nOptimization finished for:", case_name)
        print("Final trajectory shape:", xi_full.shape)
        print("History length:", len(chomp_history))

        # --------------------------------------------------
        # Convert optimized path and history back to meters
        # --------------------------------------------------
        chomp_history_m = []

        for path_w in chomp_history:
            path_m = ch.world_to_meters_xy(path_w)
            chomp_history_m.append(path_m)

        xi_full_m = xi_full.copy()
        xi_full_m[:, 0:2] = ch.world_to_meters_xy(
            xi_full[:, 0:2]
        )

        # --------------------------------------------------
        # Save final path and history data
        # --------------------------------------------------
        final_path_csv = os.path.join(
            case_output_dir,
            experiment_name + "_final_path.csv"
        )

        np.savetxt(
            final_path_csv,
            xi_full_m,
            delimiter=",",
            header="x_m,y_m,theta_rad",
            comments="",
            fmt="%.6f"
        )

        history_npy = os.path.join(
            case_output_dir,
            experiment_name + "_history.npy"
        )

        np.save(
            history_npy,
            np.array(chomp_history_m, dtype=float)
        )

        # --------------------------------------------------
        # Save final terrain image with initial/final path
        # --------------------------------------------------
        path_png = os.path.join(
            case_output_dir,
            experiment_name + "_path.png"
        )

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
                "cheap slip-risk CHOMP path"
            ],
            save_path=path_png,
            show=False
        )

        # --------------------------------------------------
        # Save GIF animation
        # --------------------------------------------------
        gif_path = os.path.join(
            case_output_dir,
            experiment_name + "_iterations.gif"
        )

        evaluator.animateChompHistoryOnTerrain(
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            chomp_history_m,
            save_path=gif_path,
            interval=250,
            show=False
        )

        # --------------------------------------------------
        # Save individual PNG frames
        # --------------------------------------------------
        frames_dir = os.path.join(
            case_output_dir,
            experiment_name + "_frames"
        )

        evaluator.saveChompHistoryFramesOnTerrain(
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            chomp_history_m,
            output_folder=frames_dir,
            frame_stride=1
        )

        # --------------------------------------------------
        # Print cheap cost comparison
        # --------------------------------------------------
        initial_cheap_cost = ch.cheap_slip_risk_cost(
            xi0,
            chomp_params.dT
        )

        final_cheap_cost = ch.cheap_slip_risk_cost(
            xi_full[:, 0:2],
            chomp_params.dT
        )

        print("\nCheap slip-risk cost:")
        print("initial:", initial_cheap_cost)
        print("final:  ", final_cheap_cost)

        print("\nSaved outputs:")
        print("path image:", path_png)
        print("gif:", gif_path)
        print("frames:", frames_dir)
        print("final path csv:", final_path_csv)
        print("history npy:", history_npy)

        plt.close("all")

    print("\nAll test cases finished.")
    print("Outputs saved in:")
    print(output_root)