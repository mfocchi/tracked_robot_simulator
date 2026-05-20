# base_controllers/chomp_slip_workspace/chomp_cost_module/terrain_geometry_cost_module.py

import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from base_controllers.chomp_slip_workspace.chomp_cost_module.base_cost_module import (
    BaseCostModule,
)


class TerrainGeometryCostModule(BaseCostModule):
    """
    Cheap terrain-geometry / slip-risk cost module.

    This cost does NOT run the dynamic simulator.

    It computes a scalar cost directly from the terrain height grid and
    the CHOMP path geometry.

    Main terms:
        - forward slope risk
        - uphill/downhill directional risk
        - lateral slope risk
        - terrain roughness risk
        - height penalty
        - curvature penalty
        - speed amplification
        - path length penalty
    """

    name = "terrain_geometry"

    def __init__(self):
        # ==================================================
        # 1) Cost weights
        # ==================================================
        self.w_forward_slope = 1.0
        self.w_lateral_slope = 5.0

        # Directional slope weights
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
        # 2) Normalization references
        # ==================================================
        self.forward_slope_ref = np.tan(np.deg2rad(15.0))
        self.lateral_slope_ref = np.tan(np.deg2rad(8.0))

        self.height_ref = 2.0
        self.roughness_ref = 0.05

        # curvature = 1 / turning_radius
        self.curvature_ref = 1.0

        # reference speed in m/s
        self.speed_ref = 0.5

        # ==================================================
        # 3) Safety thresholds
        # ==================================================
        self.height_safe = 0.5

        # ==================================================
        # 4) Terrain grid/interpolator state
        # ==================================================
        self.terrain_ready = False

        self.x_grid = None
        self.y_grid = None

        self.height_interp = None
        self.slope_x_interp = None
        self.slope_y_interp = None
        self.roughness_interp = None

        # ==================================================
        # 5) Coordinate transform state
        # ==================================================
        self.transform = None

        self.xRange = None
        self.yRange = None

        self.x_origin_m = None
        self.y_origin_m = None

        self.sx = None
        self.sy = None

    # ======================================================
    # Terrain preprocessing
    # ======================================================

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
            return_indices=True,
        )

        Z_filled = Z[tuple(nearest_indices)]

        return Z_filled

    def set_terrain_height_grid(self, X, Y, Z):
        """
        Precompute terrain height, slope, and roughness interpolators.

        Parameters
        ----------
        X, Y, Z:
            Terrain grid arrays in meters.
            Usually obtained from compute_terrain_height_grid(...).

        Notes
        -----
        RegularGridInterpolator expects points as (y, x).
        """

        Z_filled = self.fill_nan_nearest(Z)

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
            edge_order=2,
        )

        # --------------------------------------------------
        # Second derivatives: roughness approximation
        # --------------------------------------------------
        d2z_dy2, _ = np.gradient(
            dz_dy,
            y_grid,
            x_grid,
            edge_order=2,
        )

        d2z_dxdy, d2z_dx2 = np.gradient(
            dz_dx,
            y_grid,
            x_grid,
            edge_order=2,
        )

        roughness = np.sqrt(
            d2z_dx2**2
            + d2z_dy2**2
            + 2.0 * d2z_dxdy**2
        )

        # --------------------------------------------------
        # Interpolators
        # --------------------------------------------------
        self.height_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            Z_filled,
            bounds_error=False,
            fill_value=None,
        )

        self.slope_x_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            dz_dx,
            bounds_error=False,
            fill_value=None,
        )

        self.slope_y_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            dz_dy,
            bounds_error=False,
            fill_value=None,
        )

        self.roughness_interp = RegularGridInterpolator(
            (y_grid, x_grid),
            roughness,
            bounds_error=False,
            fill_value=None,
        )

        self.terrain_ready = True

        print("TerrainGeometryCostModule: terrain maps created.")
        print("x range:", x_grid[0], "to", x_grid[-1])
        print("y range:", y_grid[0], "to", y_grid[-1])

    # ======================================================
    # World-meter coordinate transform
    # ======================================================

    def set_world_meter_transform(
        self,
        transform=None,
        x_min_m=None,
        x_max_m=None,
        y_min_m=None,
        y_max_m=None,
        xRange=None,
        yRange=None,
    ):
        """
        Define conversion between CHOMP/world units and terrain meters.

        You can call this in two ways.

        Option A:
            cost_module.set_world_meter_transform(transform)

        where transform has:
            transform.world_to_meters_xy(...)
            transform.meters_to_world_xy(...)

        Option B:
            cost_module.set_world_meter_transform(
                x_min_m=...,
                x_max_m=...,
                y_min_m=...,
                y_max_m=...,
                xRange=...,
                yRange=...
            )
        """

        if transform is not None:
            if not hasattr(transform, "world_to_meters_xy"):
                raise TypeError(
                    "transform must provide world_to_meters_xy(...)."
                )

            if not hasattr(transform, "meters_to_world_xy"):
                raise TypeError(
                    "transform must provide meters_to_world_xy(...)."
                )

            self.transform = transform
            return

        if (
            x_min_m is None
            or x_max_m is None
            or y_min_m is None
            or y_max_m is None
            or xRange is None
            or yRange is None
        ):
            raise ValueError(
                "Either provide a transform object or provide "
                "x_min_m, x_max_m, y_min_m, y_max_m, xRange, yRange."
            )

        self.xRange = np.asarray(xRange, dtype=float)
        self.yRange = np.asarray(yRange, dtype=float)

        self.x_origin_m = float(x_min_m)
        self.y_origin_m = float(y_min_m)

        self.sx = (
            float(x_max_m) - float(x_min_m)
        ) / (
            self.xRange[1] - self.xRange[0]
        )

        self.sy = (
            float(y_max_m) - float(y_min_m)
        ) / (
            self.yRange[1] - self.yRange[0]
        )

    def meters_to_world_xy(self, path_m):
        """
        Convert meter coordinates to CHOMP/world coordinates.
        """

        path_m = np.asarray(path_m, dtype=float)

        if self.transform is not None:
            return self.transform.meters_to_world_xy(path_m)

        self._check_transform_ready()

        path_w = path_m.copy()
        path_w[..., 0] = (path_m[..., 0] - self.x_origin_m) / self.sx
        path_w[..., 1] = (path_m[..., 1] - self.y_origin_m) / self.sy

        return path_w

    def world_to_meters_xy(self, path_w):
        """
        Convert CHOMP/world coordinates to meter coordinates.
        """

        path_w = np.asarray(path_w, dtype=float)

        if self.transform is not None:
            return self.transform.world_to_meters_xy(path_w)

        self._check_transform_ready()

        path_m = path_w.copy()
        path_m[..., 0] = self.x_origin_m + path_w[..., 0] * self.sx
        path_m[..., 1] = self.y_origin_m + path_w[..., 1] * self.sy

        return path_m

    # ======================================================
    # Path geometry helpers
    # ======================================================

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

        speed = np.sqrt(dx**2 + dy**2)

        return speed

    def compute_path_curvature(self, path_m, dt):
        """
        Compute approximate planar curvature.

        Formula:

            kappa =
                (x_dot y_ddot - y_dot x_ddot)
                /
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

        curvature = (
            x_dot * y_ddot
            - y_dot * x_ddot
        ) / denominator

        return curvature

    def compute_path_length(self, path_m):
        """
        Compute total path length in meters.
        """

        segment_lengths = np.hypot(
            np.diff(path_m[:, 0]),
            np.diff(path_m[:, 1]),
        )

        return float(np.sum(segment_lengths))

    def compute_straight_length(self, path_m):
        """
        Compute straight-line distance between first and last waypoint.
        """

        straight_length = np.linalg.norm(
            path_m[-1, :] - path_m[0, :]
        )

        return float(max(straight_length, 1e-6))

    # ======================================================
    # Main cost computation
    # ======================================================

    def compute_cost(self, xi_xy, dt=None, **kwargs):
        """
        Compute scalar terrain-geometry cost.

        Parameters
        ----------
        xi_xy:
            CHOMP trajectory in world coordinates, shape (T, 2).

        dt:
            Time step between trajectory knots.

        Returns
        -------
        float
            Scalar terrain-geometry cost.
        """

        self._check_terrain_ready()
        self._check_transform_ready()

        if dt is None:
            raise ValueError("TerrainGeometryCostModule.compute_cost requires dt.")

        xi_xy = np.asarray(xi_xy, dtype=float)

        if xi_xy.ndim != 2:
            raise ValueError(
                f"xi_xy must be a 2D array, got shape {xi_xy.shape}."
            )

        if xi_xy.shape[1] != 2:
            raise ValueError(
                "TerrainGeometryCostModule expects xi_xy with shape (T, 2). "
                f"Got {xi_xy.shape}."
            )

        if xi_xy.shape[0] < 2:
            raise ValueError(
                "TerrainGeometryCostModule needs at least 2 waypoints."
            )

        # CHOMP/world coordinates -> meters
        path_m = self.world_to_meters_xy(xi_xy)

        x = path_m[:, 0]
        y = path_m[:, 1]

        points_yx = np.column_stack((y, x))

        # --------------------------------------------------
        # Interpolate terrain quantities along the path
        # --------------------------------------------------
        height = self._safe_array(
            self.height_interp(points_yx)
        )

        slope_x = self._safe_array(
            self.slope_x_interp(points_yx)
        )

        slope_y = self._safe_array(
            self.slope_y_interp(points_yx)
        )

        roughness = self._safe_array(
            self.roughness_interp(points_yx)
        )

        # --------------------------------------------------
        # Path direction
        # --------------------------------------------------
        theta = self.compute_path_theta(path_m)

        forward_x = np.cos(theta)
        forward_y = np.sin(theta)

        lateral_x = -np.sin(theta)
        lateral_y = np.cos(theta)

        # Terrain slope along robot forward direction
        forward_slope = (
            slope_x * forward_x
            + slope_y * forward_y
        )

        # Terrain slope along robot lateral direction
        lateral_slope = (
            slope_x * lateral_x
            + slope_y * lateral_y
        )

        # --------------------------------------------------
        # Motion geometry
        # --------------------------------------------------
        curvature = self.compute_path_curvature(
            path_m,
            dt,
        )

        speed = self.compute_path_speed(
            path_m,
            dt,
        )

        # --------------------------------------------------
        # Directional forward slope
        # --------------------------------------------------
        uphill_slope = np.maximum(0.0, forward_slope)
        downhill_slope = np.maximum(0.0, -forward_slope)

        uphill_term = (
            uphill_slope / self.forward_slope_ref
        ) ** 2

        downhill_term = (
            downhill_slope / self.forward_slope_ref
        ) ** 2

        # --------------------------------------------------
        # Lateral slope
        # --------------------------------------------------
        lateral_term = (
            lateral_slope / self.lateral_slope_ref
        ) ** 2

        # --------------------------------------------------
        # Roughness
        # --------------------------------------------------
        roughness_term = (
            roughness / self.roughness_ref
        ) ** 2

        # --------------------------------------------------
        # Curvature
        # --------------------------------------------------
        curvature_term = (
            curvature / self.curvature_ref
        ) ** 2

        # --------------------------------------------------
        # Height penalty
        # --------------------------------------------------
        height_excess = np.maximum(
            0.0,
            height - self.height_safe,
        )

        height_term = (
            height_excess / self.height_ref
        ) ** 2

        # --------------------------------------------------
        # Speed amplification
        # --------------------------------------------------
        speed_term = (
            speed / self.speed_ref
        ) ** 2

        speed_factor = 1.0 + self.w_speed * speed_term

        # --------------------------------------------------
        # Total terrain risk
        # --------------------------------------------------
        terrain_risk = (
            self.w_uphill * uphill_term
            + self.w_downhill * downhill_term
            + self.w_lateral_slope * lateral_term
            + self.w_roughness * roughness_term
            + self.w_height * height_term
        )

        turning_risk = self.w_curvature * curvature_term

        cost_density = (
            speed_factor * terrain_risk
            + turning_risk
        )

        # --------------------------------------------------
        # Path length penalty
        # --------------------------------------------------
        path_length = self.compute_path_length(path_m)
        straight_length = self.compute_straight_length(path_m)

        length_term = self.w_path_length * (
            path_length / straight_length
        ) ** 2

        total_cost = (
            np.sum(cost_density) * dt
            + length_term
        )

        return float(total_cost)

    # ======================================================
    # Internal checks/helpers
    # ======================================================

    def _safe_array(self, array):
        """
        Replace NaN and infinities with zero.

        This avoids exploding the optimizer when interpolation goes slightly
        outside valid terrain cells.
        """

        return np.nan_to_num(
            array,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _check_terrain_ready(self):
        if not self.terrain_ready:
            raise RuntimeError(
                "Terrain grid not set. "
                "Call set_terrain_height_grid(X, Y, Z) first."
            )

    def _check_transform_ready(self):
        if self.transform is not None:
            return

        if self.sx is None or self.sy is None:
            raise RuntimeError(
                "World-meter transform not set. "
                "Call set_world_meter_transform(...) first."
            )