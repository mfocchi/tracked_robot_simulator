#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robot state container used across all simulation modules.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class RobotState:
    """
    Container for robot state used across all modules.

    This provides a consistent interface regardless of whether
    we're using 2D or 3D simulation.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    theta: float = 0.0  # yaw

    # Velocities in base frame
    vx_b: float = 0.0
    vy_b: float = 0.0
    omega: float = 0.0

    @property
    def pose_2d(self) -> np.ndarray:
        """Get 2D pose [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])

    @property
    def pose_3d(self) -> np.ndarray:
        """Get 3D pose [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    @property
    def pose_6d(self) -> np.ndarray:
        """Get full 6D pose [x, y, z, roll, pitch, yaw]."""
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.theta])

    @property
    def orientation(self) -> np.ndarray:
        """Get orientation [roll, pitch, yaw]."""
        return np.array([self.roll, self.pitch, self.theta])

    @property
    def base_velocity(self) -> np.ndarray:
        """Get velocity in base frame [vx, vy]."""
        return np.array([self.vx_b, self.vy_b])

    @property
    def is_3d(self) -> bool:
        """Check if state has 3D information."""
        return self.z != 0.0 or self.roll != 0.0 or self.pitch != 0.0

    def update_from_pose(self, pose: np.ndarray, twist: Optional[np.ndarray] = None):
        """
        Update state from pose and twist arrays.

        Args:
            pose: Array [x, y, z, roll, pitch, yaw] or [x, y, yaw]
            twist: Optional array [vx, vy, vz, wx, wy, wz] or [vx, vy, omega]
        """
        if len(pose) == 6:
            self.x, self.y, self.z = pose[:3]
            self.roll, self.pitch, self.theta = pose[3:]
        elif len(pose) == 3:
            self.x, self.y, self.theta = pose
            self.z = 0.0
            self.roll = 0.0
            self.pitch = 0.0

        if twist is not None:
            if len(twist) >= 3:
                self.vx_b = twist[0]
                self.vy_b = twist[1]
                self.omega = twist[2] if len(twist) == 3 else twist[5]

    def copy(self) -> 'RobotState':
        """Create a deep copy of the state."""
        return RobotState(
            x=self.x, y=self.y, z=self.z,
            roll=self.roll, pitch=self.pitch, theta=self.theta,
            vx_b=self.vx_b, vy_b=self.vy_b, omega=self.omega
        )

    def __repr__(self) -> str:
        return (f"RobotState(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
                f"roll={self.roll:.3f}, pitch={self.pitch:.3f}, theta={self.theta:.3f})")