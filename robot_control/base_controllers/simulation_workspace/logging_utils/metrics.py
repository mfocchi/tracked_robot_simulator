#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance metrics for evaluating trajectory tracking and control.
"""
import numpy as np
from termcolor import colored


class TrackingMetrics:
    """
    Compute and store tracking performance metrics.
    """

    def __init__(self):
        # Error histories
        self.position_errors = []  # Euclidean distance error
        self.heading_errors = []  # Absolute heading error
        self.cross_track_errors = []  # Lateral error
        self.along_track_errors = []  # Longitudinal error

        # Control effort
        self.control_v = []
        self.control_omega = []

    def update(self, robot_state, des_x, des_y, des_theta, ctrl_v, ctrl_omega):
        """
        Update error metrics with current state.

        Args:
            robot_state: RobotState dataclass
            des_x, des_y, des_theta: Desired pose
            ctrl_v, ctrl_omega: Control commands
        """
        # Position error (Euclidean distance)
        e_x = des_x - robot_state.x
        e_y = des_y - robot_state.y
        pos_err = np.sqrt(e_x ** 2 + e_y ** 2)
        self.position_errors.append(pos_err)

        # Heading error (wrapped)
        e_theta = self._angle_error(des_theta, robot_state.theta)
        self.heading_errors.append(abs(e_theta))

        # Cross-track and along-track errors (in desired path frame)
        cos_des = np.cos(des_theta)
        sin_des = np.sin(des_theta)
        cross_err = -sin_des * e_x + cos_des * e_y
        along_err = cos_des * e_x + sin_des * e_y
        self.cross_track_errors.append(cross_err)
        self.along_track_errors.append(along_err)

        # Control effort
        self.control_v.append(ctrl_v)
        self.control_omega.append(ctrl_omega)

    def _angle_error(self, desired, actual):
        """Compute wrapped angle error."""
        diff = desired - actual
        return np.arctan2(np.sin(diff), np.cos(diff))

    def compute_stats(self):
        """
        Compute summary statistics.

        Returns:
            dict: Dictionary of metrics
        """
        if not self.position_errors:
            return {}

        pos = np.array(self.position_errors)
        heading = np.array(self.heading_errors)
        cross = np.array(self.cross_track_errors)
        along = np.array(self.along_track_errors)

        stats = {
            # Root Mean Square Errors
            'rmse_position': np.sqrt(np.mean(pos ** 2)),
            'rmse_heading': np.sqrt(np.mean(heading ** 2)),
            'rmse_cross_track': np.sqrt(np.mean(cross ** 2)),
            'rmse_along_track': np.sqrt(np.mean(along ** 2)),

            # Mean errors
            'mean_position': np.mean(pos),
            'mean_heading': np.mean(heading),
            'mean_cross_track': np.mean(cross),
            'mean_along_track': np.mean(along),

            # Max errors
            'max_position': np.max(pos),
            'max_heading': np.max(heading),
            'max_cross_track': np.max(np.abs(cross)),

            # Standard deviations
            'std_position': np.std(pos),
            'std_heading': np.std(heading),

            # Control effort
            'mean_abs_v': np.mean(np.abs(self.control_v)),
            'mean_abs_omega': np.mean(np.abs(self.control_omega)),
            'max_abs_v': np.max(np.abs(self.control_v)),
            'max_abs_omega': np.max(np.abs(self.control_omega)),

            # Number of samples
            'num_samples': len(pos)
        }

        return stats

    def print_report(self):
        """Print a formatted metrics report."""
        stats = self.compute_stats()
        if not stats:
            print("No data to report")
            return

        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("TRACKING PERFORMANCE REPORT", "cyan"))
        print(colored("=" * 60, "cyan"))

        print(f"\n{'Metric':<30} {'Value':>15} {'Unit':>10}")
        print("-" * 55)
        print(f"{'RMSE Position':<30} {stats['rmse_position']:>15.4f} {'m':>10}")
        print(f"{'RMSE Heading':<30} {stats['rmse_heading']:>15.4f} {'rad':>10}")
        print(f"{'RMSE Cross-Track':<30} {stats['rmse_cross_track']:>15.4f} {'m':>10}")
        print(f"{'RMSE Along-Track':<30} {stats['rmse_along_track']:>15.4f} {'m':>10}")
        print(f"{'Max Position Error':<30} {stats['max_position']:>15.4f} {'m':>10}")
        print(f"{'Max Heading Error':<30} {stats['max_heading']:>15.4f} {'rad':>10}")
        print(f"{'Max Cross-Track Error':<30} {stats['max_cross_track']:>15.4f} {'m':>10}")
        print(f"{'Mean Control |v|':<30} {stats['mean_abs_v']:>15.4f} {'m/s':>10}")
        print(f"{'Max Control |v|':<30} {stats['max_abs_v']:>15.4f} {'m/s':>10}")
        print(f"{'Num Samples':<30} {stats['num_samples']:>15d}")
        print("=" * 60)

    def reset(self):
        """Clear all stored data."""
        self.position_errors = []
        self.heading_errors = []
        self.cross_track_errors = []
        self.along_track_errors = []
        self.control_v = []
        self.control_omega = []