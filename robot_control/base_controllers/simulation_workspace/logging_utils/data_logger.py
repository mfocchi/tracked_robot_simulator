#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data logger for recording simulation state and control variables.
"""
import numpy as np
import base_controllers.params as conf


class DataLogger:
    """
    Handles all data logging during simulation.

    Stores time histories of:
    - Control commands (v, omega)
    - Desired and actual states
    - Slippage variables
    - Base pose and velocity
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim
        self._initialized = False

    def init_vars(self):
        """Initialize all logging arrays with NaN."""
        if self._initialized:
            return

        buffer_size = conf.robot_params[self.sim.robot_name]['buffer_size']
        nan_arr = np.empty(buffer_size) * np.nan

        # Control commands
        self.ctrl_v_log = nan_arr.copy()
        self.ctrl_omega_log = nan_arr.copy()
        self.v_d_log = nan_arr.copy()
        self.omega_d_log = nan_arr.copy()
        self.V_log = nan_arr.copy()
        self.V_dot_log = nan_arr.copy()

        # State logs (3 rows: x, y, theta)
        self.state_log = np.full((3, buffer_size), np.nan)
        self.des_state_log = np.full((3, buffer_size), np.nan)

        # Base pose desired (6 rows: x, y, z, roll, pitch, yaw)
        self.basePoseW_des_log = np.full((6, buffer_size), np.nan)

        # Base velocity in body frame (2 rows: vx, vy)
        self.b_base_vel_log = np.full((2, buffer_size), np.nan)

        # Slippage logs
        self.alpha_log = nan_arr.copy()
        self.beta_l_log = nan_arr.copy()
        self.beta_r_log = nan_arr.copy()
        self.alpha_control_log = nan_arr.copy()
        self.beta_l_control_log = nan_arr.copy()
        self.beta_r_control_log = nan_arr.copy()
        self.radius_log = nan_arr.copy()

        # Mirror logger-owned arrays onto the simulator so plotting and any
        # legacy-style access paths continue to work.
        self.sim.ctrl_v_log = self.ctrl_v_log
        self.sim.ctrl_omega_log = self.ctrl_omega_log
        self.sim.v_d_log = self.v_d_log
        self.sim.omega_d_log = self.omega_d_log
        self.sim.V_log = self.V_log
        self.sim.V_dot_log = self.V_dot_log
        self.sim.state_log = self.state_log
        self.sim.des_state_log = self.des_state_log
        self.sim.basePoseW_des_log = self.basePoseW_des_log
        self.sim.b_base_vel_log = self.b_base_vel_log
        self.sim.alpha_log = self.alpha_log
        self.sim.beta_l_log = self.beta_l_log
        self.sim.beta_r_log = self.beta_r_log
        self.sim.alpha_control_log = self.alpha_control_log
        self.sim.beta_l_control_log = self.beta_l_control_log
        self.sim.beta_r_control_log = self.beta_r_control_log
        self.sim.radius_log = self.radius_log

        self._initialized = True

    def log_data(self):
        """Record current simulation data at the current log counter index."""
        if not self._initialized:
            return

        idx = self.sim.log_counter
        if idx >= len(self.ctrl_v_log):
            print(f"Warning: log counter {idx} exceeds buffer size, skipping log")
            return

        # Control
        self.ctrl_v_log[idx] = self.sim.ctrl_v
        self.ctrl_omega_log[idx] = self.sim.ctrl_omega
        self.v_d_log[idx] = self.sim.v_d
        self.omega_d_log[idx] = self.sim.omega_d
        self.V_log[idx] = self.sim.V
        self.V_dot_log[idx] = self.sim.V_dot

        # Desired state
        self.des_state_log[0, idx] = self.sim.des_x
        self.des_state_log[1, idx] = self.sim.des_y
        self.des_state_log[2, idx] = self.sim.des_theta

        # Actual state (world frame)
        self.state_log[0, idx] = self.sim.basePoseW[self.sim.u.sp_crd["LX"]]
        self.state_log[1, idx] = self.sim.basePoseW[self.sim.u.sp_crd["LY"]]
        self.state_log[2, idx] = self.sim.basePoseW[self.sim.u.sp_crd["AZ"]]

        # Base pose desired (full 6D)
        self.basePoseW_des_log[:, idx] = self.sim.basePoseW_des

        # Base velocity in body frame
        self.b_base_vel_log[:, idx] = self.sim.b_base_vel

        # Slippage
        self.alpha_log[idx] = self.sim.alpha
        self.beta_l_log[idx] = self.sim.beta_l
        self.beta_r_log[idx] = self.sim.beta_r
        self.alpha_control_log[idx] = self.sim.alpha_control
        self.beta_l_control_log[idx] = self.sim.beta_l_control
        self.beta_r_control_log[idx] = self.sim.beta_r_control
        self.radius_log[idx] = self.sim.radius

    def get_valid_data(self):
        """Return data arrays cropped to valid (non-NaN) entries."""
        if not self._initialized:
            return {}

        valid = ~np.isnan(self.sim.time_log)
        if not np.any(valid):
            return {}

        return {
            'time': self.sim.time_log[valid],
            'ctrl_v': self.ctrl_v_log[valid],
            'ctrl_omega': self.ctrl_omega_log[valid],
            'v_d': self.v_d_log[valid],
            'omega_d': self.omega_d_log[valid],
            'V': self.V_log[valid],
            'V_dot': self.V_dot_log[valid],
            'state': self.state_log[:, valid],
            'des_state': self.des_state_log[:, valid],
            'base_pose_des': self.basePoseW_des_log[:, valid],
            'base_vel': self.b_base_vel_log[:, valid],
            'alpha': self.alpha_log[valid],
            'beta_l': self.beta_l_log[valid],
            'beta_r': self.beta_r_log[valid],
            'alpha_control': self.alpha_control_log[valid],
            'beta_l_control': self.beta_l_control_log[valid],
            'beta_r_control': self.beta_r_control_log[valid],
            'radius': self.radius_log[valid]
        }

    def save_to_csv(self, filename):
        """
        Save logged data to CSV file.

        Args:
            filename: Output CSV file path
        """
        import pandas as pd

        data = self.get_valid_data()
        if not data:
            print("No valid data to save")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
