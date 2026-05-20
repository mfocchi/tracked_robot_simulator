#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapter from simulation_workspace to the newer modular CHOMP implementation.
"""
import numpy as np
from matplotlib import pyplot as plt
from termcolor import colored

try:
    from base_controllers.chomp_slip_workspace.launch import chomp_launch
    from base_controllers.chomp_slip_workspace.chomp_core.chomp_config import ChompConfig
    from base_controllers.chomp_slip_workspace.chomp_utils.visual_utils import (
        compute_terrain_height_grid,
    )
    CHOMP_AVAILABLE = True
    _CHOMP_IMPORT_ERROR = None
except ImportError as base_error:
    try:
        from chomp_slip_workspace.launch import chomp_launch
        from chomp_slip_workspace.chomp_core.chomp_config import ChompConfig
        from chomp_slip_workspace.chomp_utils.visual_utils import (
            compute_terrain_height_grid,
        )
        CHOMP_AVAILABLE = True
        _CHOMP_IMPORT_ERROR = None
    except ImportError as local_error:
        CHOMP_AVAILABLE = False
        _CHOMP_IMPORT_ERROR = local_error
        print(colored(
            f"Warning: new CHOMP slip planner not available ({local_error}); straight-line fallback will be used.",
            "yellow",
        ))


class ChompPlanner:
    """
    Wrapper around the new modular CHOMP implementation in chomp_slip_workspace.

    The external interface matches the old planner so TrajectoryPlanner and the
    rest of the simulator do not need to change.
    """

    def __init__(self, sim):
        self.sim = sim
        self.last_result = None
        self._terrain_grid_cache = None
        self._terrain_grid_cache_key = None

    def get_chomp(self, start, goal, return_history=False, save_every=1):
        """
        Generate a CHOMP-optimized trajectory.

        Args:
            start: Start pose [x, y, theta]
            goal: Goal pose [x, y, theta]
            return_history: If True, return optimization history
            save_every: Kept for backward compatibility. History cadence is
                controlled by the modular CHOMP configuration.

        Returns:
            If return_history=False:
                tuple: (x_vec, y_vec, theta_vec, v_vec, omega_vec, dt)
            If return_history=True:
                tuple: (x_vec, y_vec, theta_vec, v_vec, omega_vec, dt, history)
        """
        del save_every

        if not CHOMP_AVAILABLE:
            if _CHOMP_IMPORT_ERROR is not None:
                print(colored(
                    f"New CHOMP slip planner import failed: {_CHOMP_IMPORT_ERROR}",
                    "red",
                ))
            return self._fallback_straight_line(start, goal, return_history=return_history)

        try:
            grid = self._get_terrain_grid()
            config = self._build_config()

            print(colored(
                f"Running modular CHOMP ({self._cost_name()}/{self._gradient_name()})...",
                "cyan",
            ))

            result = chomp_launch(
                start_xy_m=np.asarray(start[:2], dtype=float),
                goal_xy_m=np.asarray(goal[:2], dtype=float),
                X=grid["X"],
                Y=grid["Y"],
                Z=grid["Z"],
                x_edges=grid["x_edges"],
                y_edges=grid["y_edges"],
                cost_name=self._cost_name(),
                gradient_name=self._gradient_name(),
                config=config,
            )

            self.last_result = result
            self.sim.chomp_result = result

            output = (
                result.x,
                result.y,
                result.yaw,
                result.v,
                result.omega,
                result.dt,
            )

            print(colored(
                f"CHOMP optimization complete: {len(result.x)} waypoints",
                "green",
            ))

            if return_history:
                history_m = result.trajectory_history_m or []
                return output + (history_m,)

            return output
        except Exception as exc:
            print(colored(
                f"Modular CHOMP failed ({exc}); falling back to straight line.",
                "red",
            ))
            return self._fallback_straight_line(start, goal, return_history=return_history)

    def _build_config(self):
        """Build a ChompConfig from simulator-level settings."""
        return ChompConfig(
            dof=2,
            n_knots=int(getattr(self.sim, "CHOMP_N_KNOTS", 40)),
            dt=float(getattr(self.sim, "CHOMP_DT", 1.0)),
            max_iter=int(getattr(self.sim, "CHOMP_MAX_ITER", 100)),
            tol=float(getattr(self.sim, "CHOMP_TOL", 1.0)),
            eta=float(getattr(self.sim, "CHOMP_ETA", 0.001)),
            lambda_smooth=float(getattr(self.sim, "CHOMP_LAMBDA_SMOOTH", 200.0)),
            save_history=bool(getattr(self.sim, "CHOMP_SAVE_HISTORY", True)),
        )

    def _cost_name(self):
        return getattr(self.sim, "CHOMP_COST_NAME", "terrain_geometry")

    def _gradient_name(self):
        return getattr(self.sim, "CHOMP_GRADIENT_NAME", "spsa")

    def _get_terrain_grid(self):
        """
        Compute or reuse the terrain grid required by the modular CHOMP code.
        """
        terrain_manager = self._get_terrain_manager()
        cache_key = (
            id(terrain_manager),
            getattr(self.sim, "TERRAIN_TYPE", None),
            int(getattr(self.sim, "CHOMP_GRID_NX", 150)),
            int(getattr(self.sim, "CHOMP_GRID_NY", 150)),
            int(getattr(self.sim, "CHOMP_GRID_SAMPLES_PER_CELL", 1)),
            float(getattr(self.sim, "CHOMP_GRID_Z_MARGIN", 5.0)),
        )

        if self._terrain_grid_cache_key == cache_key and self._terrain_grid_cache is not None:
            return self._terrain_grid_cache

        X, Y, Z, x_edges, y_edges = compute_terrain_height_grid(
            terrain_manager=terrain_manager,
            nx=cache_key[2],
            ny=cache_key[3],
            samples_per_cell=cache_key[4],
            z_margin=cache_key[5],
        )

        self._terrain_grid_cache = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
        self._terrain_grid_cache_key = cache_key
        return self._terrain_grid_cache

    def _get_terrain_manager(self):
        """
        Resolve the terrain manager from the active simulator.
        """
        env_sim = getattr(self.sim, "env_simulator", None)
        if env_sim is None:
            raise RuntimeError("Environment simulator is not initialized yet.")

        terrain_manager = getattr(env_sim, "terrain_manager", None)
        if terrain_manager is not None:
            return terrain_manager

        if hasattr(env_sim, "_setup_terrain") and getattr(self.sim, "TERRAIN", False):
            env_sim._setup_terrain()
            terrain_manager = getattr(env_sim, "terrain_manager", None)

        if terrain_manager is None:
            raise RuntimeError(
                "Terrain manager is not available. The new CHOMP planner needs a terrain mesh/grid."
            )

        return terrain_manager

    def _fallback_straight_line(self, start, goal, return_history=False):
        """Generate a straight-line path when CHOMP is unavailable."""
        dt = float(getattr(self.sim, "dt", 0.001))
        duration = float(getattr(self.sim, "PLANNING_DURATION", 20.0))
        n_samples = max(2, int(duration / max(dt, 1e-6)))

        x_vec = np.linspace(start[0], goal[0], n_samples)
        y_vec = np.linspace(start[1], goal[1], n_samples)

        dx = np.gradient(x_vec)
        dy = np.gradient(y_vec)
        theta_vec = np.arctan2(dy, dx)

        v_vec = np.hypot(dx, dy) / dt
        omega_vec = np.gradient(theta_vec) / dt

        result = (x_vec, y_vec, theta_vec, v_vec, omega_vec, dt)
        if return_history:
            history = [np.column_stack((x_vec, y_vec))]
            return result + (history,)
        return result

    def plot_chomp_iterations(self, history_m=None):
        """
        Plot CHOMP optimization history.

        Args:
            history_m: Optional list of trajectories in meters. If omitted,
                planner history from the last run is used when available.
        """
        if history_m is None and self.last_result is not None:
            history_m = self.last_result.trajectory_history_m

        if not history_m:
            return

        plt.figure(figsize=(10, 8))

        for i, xi in enumerate(history_m):
            alpha = 0.3 + 0.7 * (i / max(len(history_m), 1))
            plt.plot(
                xi[:, 0],
                xi[:, 1],
                "-",
                alpha=alpha,
                label=f"Iter {i}" if i % 5 == 0 else "",
            )

        plt.plot(history_m[0][:, 0], history_m[0][:, 1], "r--", linewidth=3, label="Initial")
        plt.plot(history_m[-1][:, 0], history_m[-1][:, 1], "g-", linewidth=3, label="Final")

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("CHOMP Optimization History")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def save_obstacles_mesh(self, obstacles, output_path, height_m=2.0):
        """
        Compatibility stub kept for callers of the old planner.
        """
        del obstacles, output_path, height_m
        print(colored(
            "Obstacle mesh export is not used by the modular CHOMP planner.",
            "yellow",
        ))
