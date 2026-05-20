#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Terrain management for 3D simulations.
Provides mesh loading, projection, and ramp creation.
"""

import numpy as np
from termcolor import colored

# Re-export the original TerrainManager and ramp creation
from base_controllers.tracked_robot.simulator.terrain_manager import (
    TerrainManager,
    create_ramp_mesh
)


class TerrainHandler:
    """
    Enhanced terrain handler that wraps TerrainManager and adds
    utilities for projection and visualization.
    """

    def __init__(self, mesh_path=None, terrain_type="terrain"):
        """
        Args:
            mesh_path: Path to STL mesh file (if None, uses default)
            terrain_type: Type of terrain ('terrain', 'sphere2', 'terrain_chen')
        """
        self.terrain_type = terrain_type
        self.manager = None

        if mesh_path is not None:
            self.load_mesh(mesh_path)

    def load_mesh(self, mesh_path):
        """
        Load a terrain mesh from STL file.

        Args:
            mesh_path: Path to STL file
        """
        try:
            self.manager = TerrainManager(mesh_path)
            print(colored(f"Terrain mesh loaded: {mesh_path}", "green"))
        except Exception as e:
            print(colored(f"Failed to load terrain mesh: {e}", "red"))
            self.manager = None

    def set_mesh(self, mesh_data):
        """
        Set mesh directly from data (e.g., generated ramp).

        Args:
            mesh_data: Mesh object
        """
        if self.manager is not None:
            self.manager.set_mesh(mesh_data)

    def project_on_mesh(self, point, direction, base_yaw=0.0):
        """
        Project a 2D point onto the terrain mesh.

        Args:
            point: [x, y] 2D point
            direction: Direction vector for projection (e.g., [0,0,1])
            base_yaw: Yaw angle of robot

        Returns:
            tuple: (position_3d, roll, pitch)
        """
        if self.manager is None:
            # Flat terrain fallback
            return (
                np.array([point[0], point[1], 0.0]),
                0.0,
                0.0
            )

        return self.manager.project_on_mesh(
            point=np.array(point),
            direction=np.array(direction),
            base_yaw=base_yaw
        )

    def create_ramp(self, length=350., width=350., inclination=0.0,
                    origin=None):
        """
        Create a simple ramp terrain.

        Args:
            length: Length of ramp [m]
            width: Width of ramp [m]
            inclination: Inclination angle [rad]
            origin: Origin [x, y, z]
        """
        if origin is None:
            origin = np.array([0., 0., 0.])

        ramp_mesh = create_ramp_mesh(
            length=length,
            width=width,
            inclination=inclination,
            origin=origin
        )

        if self.manager is not None:
            self.manager.set_mesh(ramp_mesh)

        print(colored(f"Ramp created: inclination={inclination} rad", "cyan"))
        return ramp_mesh

    def get_height(self, x, y, yaw=0.0):
        """
        Get terrain height at a given (x,y) position.

        Args:
            x: X coordinate [m]
            y: Y coordinate [m]
            yaw: Yaw angle for projection

        Returns:
            float: Terrain height [m]
        """
        pos, _, _ = self.project_on_mesh(
            point=np.array([x, y]),
            direction=np.array([0., 0., 1.]),
            base_yaw=yaw
        )
        return pos[2]

    def get_surface_normal(self, x, y, yaw=0.0):
        """
        Get terrain surface normal at a point.

        Args:
            x: X coordinate [m]
            y: Y coordinate [m]
            yaw: Yaw angle for projection

        Returns:
            np.ndarray: Normal vector (3,)
        """
        _, roll, pitch = self.project_on_mesh(
            point=np.array([x, y]),
            direction=np.array([0., 0., 1.]),
            base_yaw=yaw
        )
        # Convert roll/pitch to normal vector
        normal = np.array([
            np.sin(pitch),
            -np.sin(roll) * np.cos(pitch),
            np.cos(roll) * np.cos(pitch)
        ])
        return normal / np.linalg.norm(normal)

    def is_initialized(self):
        """Check if terrain manager is available."""
        return self.manager is not None