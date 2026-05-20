#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online slippage estimator from actual robot motion.
"""
import numpy as np
import math
from base_controllers.tracked_robot.utils import maxxi_constants as constants


class SlippageEstimator:
    """
    Estimates real slippage parameters from measured robot motion.

    Computes:
    - Longitudinal slip (beta_l, beta_r): encoder velocity minus actual track velocity
    - Side slip angle (alpha): angle between velocity vector and heading
    - Turning radius
    - Body‑frame velocity
    """

    def __init__(self, sim):
        self.sim = sim
        self.u = sim.u
        self.math_utils = sim.math_utils

    def estimate(self, base_twist_w, theta, qd):
        """
        Estimate slippage from current state.

        Args:
            base_twist_w: World‑frame twist (6D or 3D)
            theta: Robot yaw angle
            qd: Wheel velocities [left, right] (rad/s)

        Returns:
            tuple: (beta_l, beta_r, alpha, radius, b_vel_xy)
        """
        wheel_l = qd[0]
        wheel_r = qd[1]

        # 1) Velocity in base frame
        b_vel_xy, omega = self._get_base_velocity(base_twist_w, theta)
        b_vel_x = b_vel_xy[0]
        v = np.linalg.norm(b_vel_xy)

        # 2) Turning radius
        radius = self._compute_radius(v, omega)

        # 3) Track velocities from encoders
        v_enc_l = constants.SPROCKET_RADIUS * wheel_l
        v_enc_r = constants.SPROCKET_RADIUS * wheel_r
        B = constants.TRACK_WIDTH

        # 4) Actual track velocities from motion
        v_track_l = b_vel_x - omega * B / 2.0
        v_track_r = b_vel_x + omega * B / 2.0

        # 5) Longitudinal slip
        beta_l = v_enc_l - v_track_l
        beta_r = v_enc_r - v_track_r

        # 6) Side slip angle
        alpha = self._compute_side_slip(b_vel_xy)

        return beta_l, beta_r, alpha, radius, b_vel_xy

    def _get_base_velocity(self, base_twist_w, theta):
        """Transform world‑frame twist to base frame."""
        if self.sim.SIMULATOR == 'distributed3d':
            w_R_b = self.math_utils.eul2Rot(self.u.angPart(self.sim.basePoseW))
            b_lin = w_R_b.T.dot(self.u.linPart(base_twist_w))
            b_ang = w_R_b.T.dot(self.u.angPart(base_twist_w))
            return b_lin[:2], b_ang[2]
        else:
            w_vel = np.array([
                base_twist_w[self.u.sp_crd["LX"]],
                base_twist_w[self.u.sp_crd["LY"]]
            ])
            omega = base_twist_w[self.u.sp_crd["AZ"]]
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            return R.T.dot(w_vel), omega

    def _compute_radius(self, v, omega):
        if abs(omega) < 1e-05 and abs(v) > 1e-05:
            return 1e08 * np.sign(v)
        elif abs(omega) < 1e-05 and abs(v) < 1e-05:
            return 1e8
        return v / omega

    def _compute_side_slip(self, b_vel_xy):
        if abs(b_vel_xy[1]) < 0.00001 or abs(b_vel_xy[0]) < 0.00001:
            return 0.0
        return math.atan2(b_vel_xy[1], b_vel_xy[0])