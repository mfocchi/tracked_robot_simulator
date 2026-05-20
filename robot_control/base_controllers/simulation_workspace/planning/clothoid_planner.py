#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clothoid curve planner for smooth path generation.
"""
import numpy as np
from termcolor import colored

try:
    import Clothoids

    CLOTHOIDS_AVAILABLE = True
except ImportError:
    CLOTHOIDS_AVAILABLE = False
    print("Warning: Clothoids library not available. Install with: pip install clothoids")


class ClothoidPlanner:
    """
    Generates smooth G1-continuous paths using Clothoid curves.

    Clothoids (Euler spirals) provide linear curvature variation,
    making them ideal for smooth vehicle trajectories.
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim

        if not CLOTHOIDS_AVAILABLE:
            print(colored(
                "Clothoids library not available. Clothoid planning disabled.",
                "red"
            ))

    def get_clothoids(self, start, goal, long_vel, dt=0.001):
        """
        Generate a Clothoid curve between start and goal poses.

        Args:
            start: Start pose [x, y, theta]
            goal: Goal pose [x, y, theta]
            long_vel: Constant longitudinal velocity [m/s]
            dt: Time step [s]

        Returns:
            tuple: (x_vec, y_vec, theta_vec, v_vec, omega_vec, dt)
        """
        if not CLOTHOIDS_AVAILABLE:
            print(colored("Clothoids not available, returning straight line", "red"))
            return self._fallback_straight_line(start, goal, long_vel, dt)

        print(colored(f"Generating Clothoid curve: start={start}, goal={goal}", "cyan"))

        # Build G1 Clothoid curve
        curve = Clothoids.ClothoidCurve("curve")
        curve.build_G1(
            start[0], start[1], start[2],
            goal[0], goal[1], goal[2]
        )

        # Compute trajectory duration
        curve_length = curve.length()
        planning_duration = curve_length / long_vel
        number_of_samples = int(np.floor(planning_duration / dt))

        if number_of_samples < 2:
            print(colored("Warning: Very short trajectory", "yellow"))
            number_of_samples = 2

        # Sample the curve
        values = np.arange(
            0, curve_length,
            curve_length / number_of_samples,
            dtype=np.float64
        )

        print(colored(
            f"Clothoid planning: length={curve_length:.2f}m, "
            f"duration={planning_duration:.2f}s, samples={number_of_samples}",
            "cyan"
        ))

        # Evaluate curve at sampled points
        xy = np.zeros((values.size, 2))
        dxdy = np.zeros((values.size, 2))
        theta = np.zeros(values.size)
        dtheta = np.zeros(values.size)

        for i in range(values.size):
            xy[i, :] = curve.eval(values[i])
            theta[i] = curve.theta(values[i])
            dxdy[i, :] = curve.eval_D(values[i])
            dtheta[i] = curve.theta_D(values[i])

        # Convert to time-based velocities
        dxdy_t = dxdy * long_vel
        omega_vec = dtheta * long_vel
        long_v_vec = np.ones(values.size) * long_vel

        return (
            xy[:, 0],  # x trajectory
            xy[:, 1],  # y trajectory
            theta,  # theta trajectory
            long_v_vec,  # velocity profile
            omega_vec,  # angular velocity profile
            dt  # time step
        )

    def _fallback_straight_line(self, start, goal, long_vel, dt):
        """Generate a simple straight line as fallback."""
        distance = np.linalg.norm(goal[:2] - start[:2])
        duration = distance / long_vel
        n_samples = max(2, int(np.floor(duration / dt)))

        x_vec = np.linspace(start[0], goal[0], n_samples)
        y_vec = np.linspace(start[1], goal[1], n_samples)
        theta_vec = np.linspace(start[2], goal[2], n_samples)
        v_vec = np.ones(n_samples) * long_vel
        omega_vec = np.zeros(n_samples)

        return x_vec, y_vec, theta_vec, v_vec, omega_vec, dt

    def compute_curvature(self, x_vec, y_vec):
        """
        Compute curvature along a path.

        Args:
            x_vec: X coordinates
            y_vec: Y coordinates

        Returns:
            np.ndarray: Curvature at each point
        """
        dx = np.gradient(x_vec)
        dy = np.gradient(y_vec)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        return curvature