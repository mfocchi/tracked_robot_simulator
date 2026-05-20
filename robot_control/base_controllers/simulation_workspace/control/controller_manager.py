#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controller manager that orchestrates different control strategies.
"""
import numpy as np
from termcolor import colored

from base_controllers.tracked_robot.utils import maxxi_constants as constants
from .lyapunov import LyapunovController, LyapunovParams, Robot
from .tracking_controller import TrackingController
from .velocity_controller import VelocityController


class ControllerManager:
    """
    Manages controller selection, initialization, and execution.

    Supports multiple control strategies:
    - CLOSED_LOOP_UNICYCLE: Standard unicycle tracking
    - CLOSED_LOOP_SLIP_AWARE: Slip-compensated tracking
    - OPEN_LOOP: Direct velocity commands (no feedback)
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim

        # Controller instances
        self.lyapunov_controller = None
        self.tracking_controller = None
        self.velocity_controller = None

        # Internal robot state
        self.robot_state = Robot()

        # Controller parameters
        self.params = None
        self.control_type = None

        # Tracking errors
        self.log_e_x = []
        self.log_e_y = []
        self.log_e_theta = []

    def setup_controller(self, control_type=None, side_slip_comp='NONE',
                         slippage_inference_type='decision_trees'):
        """
        Initialize controllers based on control type.

        Args:
            control_type: 'CLOSED_LOOP_UNICYCLE', 'CLOSED_LOOP_SLIP_AWARE', or 'OPEN_LOOP'
            side_slip_comp: 'NONE', 'MACHINE_LEARNING', or 'EXP'
            slippage_inference_type: 'decision_trees' or 'interpolator'
        """
        if control_type is None:
            control_type = self.sim.ControlType

        self.control_type = control_type
        print(colored(f"Setting up controller: {control_type}", "cyan"))

        # Setup Lyapunov parameters
        self.params = LyapunovParams(
            K_P=10.0,  # Position gain
            K_THETA=1.0,  # Orientation gain
            DT=self.sim.dt,  # Time step
            ESTIMATE_ALPHA_WITH_ACTUAL_VALUES=self.sim.ESTIMATE_ALPHA_WITH_ACTUAL_VALUES
        )

        # Initialize Lyapunov controller
        self.lyapunov_controller = LyapunovController(
            params=self.params,
            robot_constants=constants
        )

        # Configure slip compensation
        self.lyapunov_controller.setSideSlipCompensationType(side_slip_comp)
        self.lyapunov_controller.setSlippageInferenceType(slippage_inference_type)

        # Initialize tracking controller
        self.tracking_controller = TrackingController(
            self.params,
            constants,
            self.sim
        )

        # Initialize velocity controller
        self.velocity_controller = VelocityController(
            self.params,
            constants,
            self.sim
        )

        print(colored(f"Controller ready: {control_type}", "green"))

    def compute_control(self, robot_state, des_x, des_y, des_theta,
                        v_d, omega_d, v_dot_d=0.0, omega_dot_d=0.0,
                        traj_finished=False):
        """
        Compute control commands based on current state and desired trajectory.

        Args:
            robot_state: RobotState dataclass with current pose
            des_x, des_y, des_theta: Desired pose
            v_d, omega_d: Desired velocities
            v_dot_d, omega_dot_d: Desired accelerations (for slip-aware control)
            traj_finished: Whether trajectory is complete

        Returns:
            tuple: (ctrl_v, ctrl_omega, V_lyapunov, V_dot, alpha_control)
        """
        # Update internal robot state
        self._update_robot_state(robot_state)

        # Check if trajectory is finished
        if traj_finished:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Compute control based on type
        if self.control_type == 'OPEN_LOOP':
            return self._compute_open_loop(v_d, omega_d)
        elif self.control_type == 'CLOSED_LOOP_UNICYCLE':
            return self._compute_unicycle(des_x, des_y, des_theta, v_d, omega_d)
        elif self.control_type == 'CLOSED_LOOP_SLIP_AWARE':
            return self._compute_slip_aware(
                des_x, des_y, des_theta,
                v_d, omega_d, v_dot_d, omega_dot_d
            )
        else:
            print(colored(f"Unknown control type: {self.control_type}", "red"))
            return v_d, omega_d, 0.0, 0.0, 0.0

    def _update_robot_state(self, robot_state):
        """Update internal robot state from RobotState dataclass."""
        self.robot_state.x = robot_state.x
        self.robot_state.y = robot_state.y
        self.robot_state.z = robot_state.z
        self.robot_state.roll = robot_state.roll
        self.robot_state.pitch = robot_state.pitch
        self.robot_state.theta = robot_state.theta

    def _compute_open_loop(self, v_d, omega_d):
        """Open-loop control: pass through desired velocities."""
        return v_d, omega_d, 0.0, 0.0, 0.0

    def _compute_unicycle(self, des_x, des_y, des_theta, v_d, omega_d):
        """
        Standard unicycle control using Lyapunov function.

        Computes control law:
            v = v_d * cos(e_theta) + K_P * e_x
            omega = omega_d + K_THETA * e_theta + v_d * sinc(e_theta) * e_y
        """
        ctrl_v, ctrl_omega, V, V_dot = self.lyapunov_controller.control_unicycle(
            self.robot_state,
            self.sim.time,
            des_x, des_y, des_theta,
            v_d, omega_d,
            False  # traj_finished
        )

        # Log tracking errors
        self._log_errors(des_x, des_y, des_theta)

        return ctrl_v, ctrl_omega, V, V_dot, 0.0

    def _compute_slip_aware(self, des_x, des_y, des_theta,
                            v_d, omega_d, v_dot_d, omega_dot_d):
        """
        Slip-aware control with alpha (side-slip) compensation.

        This controller:
        1. Estimates expected side-slip angle
        2. Compensates desired heading
        3. Applies Lyapunov control with compensated heading
        """
        # Get alpha prediction model if available
        model_alpha = None
        if hasattr(self.sim, 'compensator') and self.sim.compensator is not None:
            if hasattr(self.sim.compensator, 'models'):
                model_alpha = self.sim.compensator.models.get('alpha')

        ctrl_v, ctrl_omega, V, V_dot, alpha_control = \
            self.lyapunov_controller.control_alpha(
                self.robot_state,
                self.sim.time,
                des_x, des_y, des_theta,
                v_d, omega_d,
                v_dot_d, omega_dot_d,
                False,  # traj_finished
                model_alpha,
                approx=True
            )

        # Log tracking errors
        self._log_errors(des_x, des_y, des_theta - alpha_control)

        return ctrl_v, ctrl_omega, V, V_dot, alpha_control

    def _log_errors(self, des_x, des_y, des_theta):
        """Log tracking errors."""
        e_x = des_x - self.robot_state.x
        e_y = des_y - self.robot_state.y
        e_theta = self._angle_error(des_theta, self.robot_state.theta)

        self.log_e_x.append(e_x)
        self.log_e_y.append(e_y)
        self.log_e_theta.append(e_theta)

    def _angle_error(self, desired, actual):
        """Compute wrapped angle error."""
        error = desired - actual
        return np.arctan2(np.sin(error), np.cos(error))

    def get_errors(self):
        """
        Get tracking error history.

        Returns:
            tuple: (log_e_x, log_e_y, log_e_theta) arrays
        """
        return (
            np.array(self.log_e_x),
            np.array(self.log_e_y),
            np.array(self.log_e_theta)
        )

    def getErrors(self):
        """Compatibility alias for the legacy camelCase controller API."""
        return self.get_errors()

    def compute_rmse(self):
        """Compute Root Mean Square Error for tracking."""
        if not self.log_e_x:
            return 0.0, 0.0, 0.0

        e_x = np.array(self.log_e_x)
        e_y = np.array(self.log_e_y)
        e_theta = np.array(self.log_e_theta)

        # Position error norm
        e_xy = np.sqrt(e_x ** 2 + e_y ** 2)
        rmse_xy = np.sqrt(np.mean(e_xy ** 2))
        rmse_theta = np.sqrt(np.mean(e_theta ** 2))

        return rmse_xy, rmse_theta, np.mean(e_xy)

    def reset_errors(self):
        """Reset tracking error logs."""
        self.log_e_x = []
        self.log_e_y = []
        self.log_e_theta = []

    def get_control_info(self):
        """Get information about current controller configuration."""
        return {
            'control_type': self.control_type,
            'params': {
                'K_P': self.params.K_P,
                'K_THETA': self.params.K_THETA,
                'DT': self.params.DT
            },
            'robot_state': {
                'x': self.robot_state.x,
                'y': self.robot_state.y,
                'theta': self.robot_state.theta
            }
        }

    def update_gains(self, K_P=None, K_THETA=None):
        """
        Update controller gains online.

        Args:
            K_P: New position gain (or None to keep current)
            K_THETA: New orientation gain (or None to keep current)
        """
        if K_P is not None:
            self.params.K_P = K_P
            print(colored(f"Updated K_P to {K_P}", "yellow"))

        if K_THETA is not None:
            self.params.K_THETA = K_THETA
            print(colored(f"Updated K_THETA to {K_THETA}", "yellow"))
