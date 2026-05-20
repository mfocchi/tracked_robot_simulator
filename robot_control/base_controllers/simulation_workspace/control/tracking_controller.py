#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory tracking controller with various control laws.
"""
import numpy as np
from termcolor import colored


class TrackingController:
    """
    Implements various trajectory tracking control laws.

    Supports:
    - Pure pursuit
    - Stanley controller
    - Nonlinear Lyapunov-based tracking
    - Model predictive control (placeholder)
    """

    def __init__(self, params, constants, sim):
        """
        Args:
            params: LyapunovParams instance
            constants: Robot constants
            sim: Reference to GenericSimulator instance
        """
        self.params = params
        self.constants = constants
        self.sim = sim

        # Control gains
        self.K_p = params.K_P  # Position gain
        self.K_theta = params.K_THETA  # Orientation gain

        # Look-ahead distance for pure pursuit
        self.look_ahead_distance = 1.0

    def pure_pursuit(self, robot_state, path_x, path_y):
        """
        Pure pursuit controller.

        Finds a look-ahead point on the path and computes steering
        to track that point.

        Args:
            robot_state: Current robot state
            path_x: Path x coordinates
            path_y: Path y coordinates

        Returns:
            tuple: (v, omega) control commands
        """
        # Find closest point on path
        distances = np.sqrt(
            (path_x - robot_state.x) ** 2 +
            (path_y - robot_state.y) ** 2
        )
        closest_idx = np.argmin(distances)

        # Find look-ahead point
        look_ahead_idx = closest_idx
        cumulative_dist = 0.0

        for i in range(closest_idx, len(path_x) - 1):
            segment_dist = np.sqrt(
                (path_x[i + 1] - path_x[i]) ** 2 +
                (path_y[i + 1] - path_y[i]) ** 2
            )
            cumulative_dist += segment_dist

            if cumulative_dist >= self.look_ahead_distance:
                look_ahead_idx = i + 1
                break

        # Get look-ahead point
        look_ahead_x = path_x[look_ahead_idx]
        look_ahead_y = path_y[look_ahead_idx]

        # Transform to robot frame
        dx = look_ahead_x - robot_state.x
        dy = look_ahead_y - robot_state.y

        # Transform to body frame
        cos_theta = np.cos(robot_state.theta)
        sin_theta = np.sin(robot_state.theta)

        dx_body = cos_theta * dx + sin_theta * dy
        dy_body = -sin_theta * dx + cos_theta * dy

        # Compute curvature
        curvature = 2.0 * dy_body / (self.look_ahead_distance ** 2)

        # Compute control
        v = self.params.v_d  # Maintain desired speed
        omega = v * curvature

        return v, omega

    def stanley_controller(self, robot_state, path_x, path_y, path_theta):
        """
        Stanley controller for path tracking.

        Combines heading error and cross-track error.

        Args:
            robot_state: Current robot state
            path_x: Path x coordinates
            path_y: Path y coordinates
            path_theta: Path heading angles

        Returns:
            tuple: (v, omega) control commands
        """
        # Find closest point on path
        distances = np.sqrt(
            (path_x - robot_state.x) ** 2 +
            (path_y - robot_state.y) ** 2
        )
        closest_idx = np.argmin(distances)

        # Heading error
        heading_error = self._normalize_angle(
            path_theta[closest_idx] - robot_state.theta
        )

        # Cross-track error
        dx = robot_state.x - path_x[closest_idx]
        dy = robot_state.y - path_y[closest_idx]

        # Transform to path frame
        cos_path = np.cos(path_theta[closest_idx])
        sin_path = np.sin(path_theta[closest_idx])

        cross_track_error = -sin_path * dx + cos_path * dy

        # Stanley control law
        k = 0.5  # Gain
        v = self.params.v_d  # Desired speed

        # Steering angle
        steering = heading_error + np.arctan2(
            k * cross_track_error,
            max(v, 0.1)  # Avoid division by zero
        )

        omega = v * np.tan(steering) / self.constants.TRACK_WIDTH

        return v, omega

    def lyapunov_tracking(self, robot_state, des_x, des_y, des_theta, v_d, omega_d):
        """
        Lyapunov-based nonlinear tracking controller.

        Uses a Lyapunov function to prove stability and derive
        control law.

        Args:
            robot_state: Current robot state
            des_x, des_y, des_theta: Desired pose
            v_d, omega_d: Desired velocities

        Returns:
            tuple: (v, omega, V, V_dot) control and Lyapunov values
        """
        # Compute errors in world frame
        e_x = des_x - robot_state.x
        e_y = des_y - robot_state.y
        e_theta = self._normalize_angle(des_theta - robot_state.theta)

        # Transform errors to body frame
        cos_theta = np.cos(robot_state.theta)
        sin_theta = np.sin(robot_state.theta)

        e_x_body = cos_theta * e_x + sin_theta * e_y
        e_y_body = -sin_theta * e_x + cos_theta * e_y

        # Lyapunov function
        V = 0.5 * (e_x_body ** 2 + e_y_body ** 2) + \
            0.5 * (1 - np.cos(e_theta)) ** 2

        # Control law
        if abs(e_theta) < 1e-06:
            sinc_e_theta = 1.0
        else:
            sinc_e_theta = np.sin(e_theta) / e_theta

        v = v_d * np.cos(e_theta) + self.K_p * e_x_body
        omega = omega_d + self.K_theta * e_theta + \
                v_d * sinc_e_theta * e_y_body

        # Lyapunov derivative
        V_dot = -self.K_p * e_x_body ** 2 - \
                self.K_theta * e_theta ** 2

        return v, omega, V, V_dot

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def compute_look_ahead_point(self, robot_state, path, look_ahead_dist):
        """
        Compute look-ahead point on a path.

        Args:
            robot_state: Current robot state
            path: Nx2 array of [x, y] path points
            look_ahead_dist: Look-ahead distance

        Returns:
            np.ndarray: [x, y] look-ahead point
        """
        # Find closest point
        distances = np.sqrt(
            (path[:, 0] - robot_state.x) ** 2 +
            (path[:, 1] - robot_state.y) ** 2
        )
        closest_idx = np.argmin(distances)

        # Search forward for look-ahead point
        for i in range(closest_idx, len(path) - 1):
            dist_to_point = np.sqrt(
                (path[i, 0] - robot_state.x) ** 2 +
                (path[i, 1] - robot_state.y) ** 2
            )

            if dist_to_point >= look_ahead_dist:
                return path[i]

        # Return last point if no look-ahead found
        return path[-1]