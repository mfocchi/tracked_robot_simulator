#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Obstacle generation for simulation environments.
"""

import numpy as np
from termcolor import colored


class ObstacleGenerator:
    """
    Generates obstacle definitions for planners and visualization.
    """

    def __init__(self):
        self.obstacles = []

    def add_rectangle(self, center_x, center_y, width, height, label=""):
        """
        Add a rectangular obstacle.

        Args:
            center_x, center_y: Center position
            width: Width in x direction
            height: Height in y direction
            label: Optional label
        """
        half_w = width / 2.0
        half_h = height / 2.0

        obstacle = {
            "X": np.array([
                center_x - half_w,
                center_x + half_w,
                center_x + half_w,
                center_x - half_w
            ]),
            "Y": np.array([
                center_y - half_h,
                center_y - half_h,
                center_y + half_h,
                center_y + half_h
            ]),
            "label": label
        }
        self.obstacles.append(obstacle)

    def add_triangle(self, x1, y1, x2, y2, x3, y3, label=""):
        """
        Add a triangular obstacle.

        Args:
            x1,y1, x2,y2, x3,y3: Vertices of triangle
            label: Optional label
        """
        obstacle = {
            "X": np.array([x1, x2, x3]),
            "Y": np.array([y1, y2, y3]),
            "label": label
        }
        self.obstacles.append(obstacle)

    def add_circle(self, center_x, center_y, radius, num_points=20, label=""):
        """
        Approximate a circular obstacle with a polygon.

        Args:
            center_x, center_y: Center position
            radius: Circle radius
            num_points: Number of vertices for approximation
            label: Optional label
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)

        obstacle = {
            "X": x,
            "Y": y,
            "label": label
        }
        self.obstacles.append(obstacle)

    def add_default_obstacles(self):
        """
        Add default set of obstacles for testing.
        """
        self.add_rectangle(150, 50, 200, 100, "building1")
        self.add_triangle(200, 300, 300, 300, 250, 400, "rock1")

    def get_obstacle_list(self):
        """
        Get obstacles in the format expected by CHOMP planner.

        Returns:
            list of dicts with 'X' and 'Y' keys
        """
        return self.obstacles

    def clear(self):
        """Remove all obstacles."""
        self.obstacles = []

    def transform_to_world(self, world_origin, scale):
        """
        Transform obstacles to world coordinate frame.

        Args:
            world_origin: [x0, y0] origin offset
            scale: Scale factor
        """
        transformed = []
        for obs in self.obstacles:
            new_obs = {
                "X": (obs["X"] - world_origin[0]) * scale,
                "Y": (obs["Y"] - world_origin[1]) * scale,
                "label": obs.get("label", "")
            }
            transformed.append(new_obs)
        return transformed

    def export_to_stl(self, filename, height=2.0):
        """
        Export obstacles as STL mesh for visualization.

        Args:
            filename: Output STL file path
            height: Extrusion height
        """
        try:
            from stl import mesh
            import os

            all_meshes = []
            for obs in self.obstacles:
                # Create extrusion for each polygon
                x = obs["X"]
                y = obs["Y"]
                n = len(x)

                # Create bottom and top faces (simplified)
                for i in range(n):
                    # This is a placeholder - proper STL generation would be more complex
                    pass

            print(colored(f"Obstacles exported to {filename}", "green"))
        except ImportError:
            print(colored("numpy-stl not installed. Install with: pip install numpy-stl", "yellow"))

    def plot_obstacles(self, ax):
        """
        Plot obstacles on a matplotlib axis.

        Args:
            ax: Matplotlib axis
        """
        for obs in self.obstacles:
            x = np.append(obs["X"], obs["X"][0])
            y = np.append(obs["Y"], obs["Y"][0])
            ax.fill(x, y, alpha=0.3, color='red', label=obs.get("label", ""))
            ax.plot(x, y, 'k-', linewidth=2)