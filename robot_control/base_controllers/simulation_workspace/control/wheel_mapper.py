#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wheel velocity mapping and coordinate transformations.
"""
import numpy as np
from sensor_msgs.msg import JointState
from base_controllers.tracked_robot.utils import maxxi_constants as constants


class WheelMapper:
    """
    Handles velocity-to-wheel and wheel-to-velocity conversions.

    Maps between:
    - Base frame velocities (v, omega)
    - Individual wheel velocities (qd_left, qd_right)
    - Track velocities (v_left, v_right)
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim
        self.track_width = constants.TRACK_WIDTH
        self.sprocket_radius = constants.SPROCKET_RADIUS
        self.max_speed = constants.MAXSPEED_RADS_PULLEY

    def map_to_wheels(self, v_des, omega_des, project_to_terrain=False):
        """
        Convert desired base velocity to wheel velocities.

        Args:
            v_des: Desired linear velocity in base frame [m/s]
            omega_des: Desired angular velocity in base frame [rad/s]
            project_to_terrain: If True, project velocity onto terrain

        Returns:
            np.ndarray: [qd_left, qd_right] wheel velocities [rad/s]
        """
        qd_des = np.zeros(2)

        # Convert base velocity to track velocities
        v_left = v_des - omega_des * self.track_width / 2.0
        v_right = v_des + omega_des * self.track_width / 2.0

        # Convert track velocities to wheel angular velocities
        qd_des[0] = v_left / self.sprocket_radius  # Left wheel
        qd_des[1] = v_right / self.sprocket_radius  # Right wheel

        # Apply speed limits
        qd_des = self._apply_speed_limits(qd_des)

        return qd_des

    def map_from_wheels(self, wheel_l, wheel_r):
        """
        Convert wheel velocities to base velocity.

        Args:
            wheel_l: Left wheel velocity [rad/s] (scalar or array)
            wheel_r: Right wheel velocity [rad/s] (scalar or array)

        Returns:
            tuple: (v, omega) base velocities [m/s, rad/s]
        """
        if not np.isscalar(wheel_l):
            # Handle array inputs
            v = np.zeros_like(wheel_l)
            omega = np.zeros_like(wheel_l)
            for i in range(len(wheel_l)):
                v[i] = self.sprocket_radius * (wheel_l[i] + wheel_r[i]) / 2.0
                omega[i] = self.sprocket_radius / self.track_width * \
                           (wheel_r[i] - wheel_l[i])
            return v, omega

        # Handle scalar inputs
        v = self.sprocket_radius * (wheel_l + wheel_r) / 2.0
        omega = self.sprocket_radius / self.track_width * (wheel_r - wheel_l)

        return v, omega

    def map_to_track_velocities(self, qd_wheels):
        """
        Convert wheel angular velocities to track linear velocities.

        Args:
            qd_wheels: [qd_left, qd_right] wheel velocities [rad/s]

        Returns:
            tuple: (v_track_l, v_track_r) track velocities [m/s]
        """
        v_track_l = self.sprocket_radius * qd_wheels[0]
        v_track_r = self.sprocket_radius * qd_wheels[1]

        return v_track_l, v_track_r

    def compute_actual_track_velocities(self, b_vel_x, omega):
        """
        Compute actual track velocities from base motion.

        Args:
            b_vel_x: Longitudinal velocity in base frame [m/s]
            omega: Angular velocity [rad/s]

        Returns:
            tuple: (v_track_l, v_track_r) actual track velocities [m/s]
        """
        v_track_l = b_vel_x - omega * self.track_width / 2.0
        v_track_r = b_vel_x + omega * self.track_width / 2.0

        return v_track_l, v_track_r

    def compute_turning_radius(self, v, omega):
        """
        Compute turning radius from base velocities.

        Args:
            v: Linear velocity [m/s]
            omega: Angular velocity [rad/s]

        Returns:
            float: Turning radius [m] (positive = left turn, negative = right turn)
        """
        if abs(omega) < 1e-05:
            if abs(v) > 1e-05:
                return 1e08 * np.sign(v)  # Nearly straight
            else:
                return 1e08  # Stationary

        return v / omega

    def project_velocity_on_terrain(self, v_des, omega_des, euler):
        """
        Project desired velocity onto terrain for 3D simulation.

        Args:
            v_des: Desired velocity in world frame [m/s]
            omega_des: Desired angular velocity in world frame [rad/s]
            euler: Current Euler angles [roll, pitch, yaw]

        Returns:
            tuple: (v_projected, omega_projected) in base frame
        """
        w_R_b = self.sim.math_utils.eul2Rot(euler)
        hf_R_b = self.sim.math_utils.eul2Rot(np.array([euler[0], euler[1], 0.0]))

        # Project linear velocity onto horizontal plane of base frame
        v_projected = hf_R_b[0].dot(np.array([v_des, 0.0, 0.0]))

        # Project angular velocity onto vertical axis of base frame
        omega_projected = w_R_b[2].dot(np.array([0.0, 0.0, omega_des]))

        return v_projected, omega_projected

    def _apply_speed_limits(self, qd_des):
        """Apply wheel speed limits."""
        qd_limited = np.clip(qd_des, -self.max_speed, self.max_speed)

        if np.any(qd_limited != qd_des):
            from termcolor import colored
            print(colored(
                f"Warning: Wheel speed limited from {qd_des} to {qd_limited}",
                "yellow"
            ))

        return qd_limited

    def create_joint_state_msg(self, q_des, qd_des, tau_ffwd, joint_names, stamp):
        """
        Create a JointState ROS message.

        Args:
            q_des: Desired joint positions
            qd_des: Desired joint velocities
            tau_ffwd: Feed-forward torques
            joint_names: List of joint names
            stamp: ROS timestamp

        Returns:
            JointState message
        """
        msg = JointState()
        msg.name = joint_names
        msg.header.stamp = stamp
        msg.position = q_des
        msg.velocity = qd_des
        msg.effort = tau_ffwd

        return msg

    def compute_differential(self, v_des, omega_des, dt):
        """
        Compute differential drive kinematics for one step.

        Args:
            v_des: Linear velocity [m/s]
            omega_des: Angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            tuple: (dx, dy, dtheta) pose change in body frame
        """
        if abs(omega_des) < 1e-06:
            # Straight motion
            dx = v_des * dt
            dy = 0.0
            dtheta = 0.0
        else:
            # Turning motion
            R = v_des / omega_des  # Turning radius
            dtheta = omega_des * dt
            dx = R * np.sin(dtheta)
            dy = R * (1 - np.cos(dtheta))

        return dx, dy, dtheta

    def compute_icc(self, v_des, omega_des):
        """
        Compute Instantaneous Center of Curvature (ICC).

        Args:
            v_des: Linear velocity [m/s]
            omega_des: Angular velocity [rad/s]

        Returns:
            tuple: (icc_x, icc_y) ICC position in body frame, or None if straight
        """
        if abs(omega_des) < 1e-06:
            return None  # No ICC for straight motion

        # ICC is on the y-axis of the body frame
        icc_x = 0.0
        icc_y = v_des / omega_des  # Distance from center to ICC

        return icc_x, icc_y