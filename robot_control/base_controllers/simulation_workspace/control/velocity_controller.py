#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Velocity controller for smooth velocity tracking.
"""
import numpy as np
from termcolor import colored


class VelocityController:
    """
    Low-level velocity controller with acceleration limits.

    Provides:
    - Velocity smoothing
    - Acceleration limiting
    - Anti-windup for integral terms
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

        # Velocity limits
        self.max_linear_velocity = 1.0  # [m/s]
        self.max_angular_velocity = 1.0  # [rad/s]
        self.max_linear_accel = 2.0  # [m/s^2]
        self.max_angular_accel = 4.0  # [rad/s^2]

        # Previous velocity commands (for acceleration limiting)
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # Integral terms
        self.integral_v = 0.0
        self.integral_omega = 0.0
        self.max_integral = 0.5

    def smooth_velocity(self, v_des, omega_des):
        """
        Apply velocity smoothing with acceleration limits.

        Args:
            v_des: Desired linear velocity [m/s]
            omega_des: Desired angular velocity [rad/s]

        Returns:
            tuple: (v_smooth, omega_smooth) smoothed velocities
        """
        dt = self.params.DT

        # Limit linear acceleration
        dv = v_des - self.prev_v
        dv_limited = np.clip(dv,
                             -self.max_linear_accel * dt,
                             self.max_linear_accel * dt)
        v_smooth = self.prev_v + dv_limited

        # Limit angular acceleration
        domega = omega_des - self.prev_omega
        domega_limited = np.clip(domega,
                                 -self.max_angular_accel * dt,
                                 self.max_angular_accel * dt)
        omega_smooth = self.prev_omega + domega_limited

        # Apply velocity limits
        v_smooth = np.clip(v_smooth, -self.max_linear_velocity, self.max_linear_velocity)
        omega_smooth = np.clip(omega_smooth, -self.max_angular_velocity, self.max_angular_velocity)

        # Store for next iteration
        self.prev_v = v_smooth
        self.prev_omega = omega_smooth

        return v_smooth, omega_smooth

    def velocity_pid(self, v_des, v_actual, omega_des, omega_actual):
        """
        PID velocity controller.

        Args:
            v_des: Desired linear velocity [m/s]
            v_actual: Actual linear velocity [m/s]
            omega_des: Desired angular velocity [rad/s]
            omega_actual: Actual angular velocity [rad/s]

        Returns:
            tuple: (v_cmd, omega_cmd) velocity commands
        """
        dt = self.params.DT

        # PID gains
        Kp_v = 1.0
        Ki_v = 0.1
        Kd_v = 0.01

        Kp_omega = 1.0
        Ki_omega = 0.1
        Kd_omega = 0.01

        # Linear velocity PID
        error_v = v_des - v_actual
        self.integral_v += error_v * dt
        self.integral_v = np.clip(self.integral_v, -self.max_integral, self.max_integral)
        derivative_v = (error_v - getattr(self, 'prev_error_v', 0)) / dt

        v_cmd = Kp_v * error_v + Ki_v * self.integral_v + Kd_v * derivative_v
        self.prev_error_v = error_v

        # Angular velocity PID
        error_omega = omega_des - omega_actual
        self.integral_omega += error_omega * dt
        self.integral_omega = np.clip(self.integral_omega, -self.max_integral, self.max_integral)
        derivative_omega = (error_omega - getattr(self, 'prev_error_omega', 0)) / dt

        omega_cmd = Kp_omega * error_omega + Ki_omega * self.integral_omega + Kd_omega * derivative_omega
        self.prev_error_omega = error_omega

        # Apply smoothing
        v_cmd, omega_cmd = self.smooth_velocity(v_cmd, omega_cmd)

        return v_cmd, omega_cmd

    def compute_feedforward(self, v_des, omega_des, v_dot_d, omega_dot_d):
        """
        Compute feed-forward velocity commands with dynamics compensation.

        Args:
            v_des: Desired linear velocity [m/s]
            omega_des: Desired angular velocity [rad/s]
            v_dot_d: Desired linear acceleration [m/s^2]
            omega_dot_d: Desired angular acceleration [rad/s^2]

        Returns:
            tuple: (v_ff, omega_ff) feed-forward commands
        """
        # Simple feed-forward with inertia compensation
        # In a real system, this would use the inverse dynamics model

        # Track inertia parameters (example values)
        track_mass = 100.0  # kg
        track_inertia = 10.0  # kg*m^2
        damping = 10.0  # N*s/m

        # Feed-forward linear velocity
        v_ff = v_des + (track_mass / damping) * v_dot_d

        # Feed-forward angular velocity
        omega_ff = omega_des + (track_inertia / (damping * self.constants.TRACK_WIDTH ** 2)) * omega_dot_d

        return v_ff, omega_ff

    def reset_integrators(self):
        """Reset integral terms (anti-windup)."""
        self.integral_v = 0.0
        self.integral_omega = 0.0

    def set_velocity_limits(self, v_max=None, omega_max=None):
        """
        Update velocity limits.

        Args:
            v_max: Maximum linear velocity [m/s]
            omega_max: Maximum angular velocity [rad/s]
        """
        if v_max is not None:
            self.max_linear_velocity = v_max
            print(colored(f"Max linear velocity set to {v_max} m/s", "yellow"))

        if omega_max is not None:
            self.max_angular_velocity = omega_max
            print(colored(f"Max angular velocity set to {omega_max} rad/s", "yellow"))

    def set_accel_limits(self, accel_max=None, omega_accel_max=None):
        """
        Update acceleration limits.

        Args:
            accel_max: Maximum linear acceleration [m/s^2]
            omega_accel_max: Maximum angular acceleration [rad/s^2]
        """
        if accel_max is not None:
            self.max_linear_accel = accel_max
            print(colored(f"Max linear acceleration set to {accel_max} m/s^2", "yellow"))

        if omega_accel_max is not None:
            self.max_angular_accel = omega_accel_max
            print(colored(f"Max angular acceleration set to {omega_accel_max} rad/s^2", "yellow"))