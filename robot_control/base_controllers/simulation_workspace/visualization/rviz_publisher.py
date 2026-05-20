#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RViz visualization helpers that wrap the legacy ros_pub object defensively.
"""
import numpy as np
from termcolor import colored

try:
    import rospkg
except ModuleNotFoundError:
    rospkg = None


class RvizPublisher:
    """
    Thin wrapper around the raw ros_pub object from the original controller.

    The modular code should be able to run even when the exact ros_pub helper
    implementation differs slightly or is not yet initialized.
    """

    def __init__(self, sim):
        self.sim = sim

    @property
    def ros_pub(self):
        return getattr(self.sim, "ros_pub", None)

    def _call_first_supported(self, method_specs):
        ros_pub = self.ros_pub
        if ros_pub is None:
            return False

        for method_name, args, kwargs in method_specs:
            method = getattr(ros_pub, method_name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
                return True
            except TypeError:
                continue
        return False

    def add_marker(self, position, color="white", radius=0.5, alpha=1.0):
        specs = [
            ("add_marker", (position,), {"color": color, "radius": radius, "alpha": alpha}),
            ("add_sphere", (position,), {"radius": radius, "color": color, "alpha": alpha}),
        ]
        self._call_first_supported(specs)

    def add_arrow(self, start, end, color="white", alpha=1.0):
        specs = [
            ("add_arrow", (start, end), {"color": color, "alpha": alpha}),
        ]
        self._call_first_supported(specs)

    def add_mesh(
            self,
            package_name,
            mesh_path,
            position=(0, 0, 0),
            orientation=None,
            color="red",
            alpha=0.5,
    ):
        """
        Publish an STL/mesh marker through the legacy ros_pub object.

        The original tracked_robot_simulator ros_pub.add_mesh signature is usually:

            add_mesh(package_name, mesh_path, position=..., color=..., alpha=...)

        It does not necessarily accept orient/orientation.
        """

        position = np.array(position)

        specs = [
            # Most important: legacy signature used in professor's original code
            (
                "add_mesh",
                (package_name, mesh_path),
                {"position": position, "color": color, "alpha": alpha},
            ),

            # Fallback: positional position
            (
                "add_mesh",
                (package_name, mesh_path, position),
                {"color": color, "alpha": alpha},
            ),

            # Optional fallback if some version supports orientation
            (
                "add_mesh",
                (package_name, mesh_path),
                {
                    "position": position,
                    "orientation": orientation,
                    "color": color,
                    "alpha": alpha,
                },
            ),
        ]

        if not self._call_first_supported(specs):
            print(colored(
                f"ros_pub could not publish mesh: package={package_name}, mesh={mesh_path}",
                "yellow",
            ))

    def add_plane(self, pos, orient=(0, 0, 0), color="white", alpha=1.0):
        specs = [
            ("add_plane", (), {"pos": pos, "orient": orient, "color": color, "alpha": alpha}),
            ("add_plane", (pos,), {"orient": orient, "color": color, "alpha": alpha}),
        ]
        self._call_first_supported(specs)

    def publish_terrain_and_obstacles(self):
        if self.sim.TERRAIN and hasattr(self.sim, "TERRAIN_TYPE"):
            mesh_file = f"/meshes/{self.sim.TERRAIN_TYPE}.stl"
            self.add_mesh(
                "tractor_description",
                mesh_file,
                position=np.array([0.0, 0.0, 0.0]),
                color="red",
                alpha=1.0,
            )

    def publish_visuals(self, delete_markers=False):
        specs = [
            ("publishVisual", (), {"delete_markers": delete_markers}),
            ("publish_visuals", (), {"delete_markers": delete_markers}),
        ]
        self._call_first_supported(specs)

    def publishVisual(self, delete_markers=False):
        self.publish_visuals(delete_markers=delete_markers)

    def clear_markers(self):
        specs = [
            ("clear_markers", (), {}),
            ("delete_all_markers", (), {}),
        ]
        self._call_first_supported(specs)
