#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS bag recorder for capturing simulation data.
"""
import rospy as ros
from termcolor import colored

from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder


class RosbagRecorder:
    """
    Wrapper around RosbagControlledRecorder for easy bag management.
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim
        self.recorder = None
        self.is_recording = False

    def start_recording(self, bag_name=None, topics=None):
        """
        Start recording a ROS bag.

        Args:
            bag_name: Name of the bag file (auto-generated if None)
            topics: List of topics to record (records all if None)
        """
        if self.is_recording:
            print(colored("Already recording", "yellow"))
            return

        if bag_name is None:
            bag_name = self._generate_bag_name()

        try:
            self.recorder = RosbagControlledRecorder(bag_name=bag_name)
            self.recorder.start_recording_srv()
            self.is_recording = True
            print(colored(f"Started recording to {bag_name}", "green"))
        except Exception as e:
            print(colored(f"Failed to start recording: {e}", "red"))

    def stop_recording(self):
        """Stop the current recording."""
        if not self.is_recording:
            return

        try:
            self.recorder.stop_recording_srv()
            self.is_recording = False
            print(colored("Recording stopped", "green"))
        except Exception as e:
            print(colored(f"Failed to stop recording: {e}", "red"))

    def _generate_bag_name(self):
        """Automatically generate a descriptive bag name based on simulator settings."""
        parts = []

        # Control type
        if hasattr(self.sim, 'ControlType'):
            parts.append(self.sim.ControlType)

        # Slip compensation
        if hasattr(self.sim, 'LONG_SLIP_COMPENSATION'):
            parts.append(f"Long_{self.sim.LONG_SLIP_COMPENSATION}")
        if hasattr(self.sim, 'SIDE_SLIP_COMPENSATION'):
            parts.append(f"Side_{self.sim.SIDE_SLIP_COMPENSATION}")

        # Friction
        if hasattr(self.sim, 'friction_coefficient'):
            parts.append(f"fr_{self.sim.friction_coefficient}")

        # Simulator type
        if hasattr(self.sim, 'SIMULATOR'):
            parts.append(self.sim.SIMULATOR)

        # For open loop identification
        if self.sim.ControlType == 'OPEN_LOOP':
            if hasattr(self.sim, 'IDENT_TYPE') and self.sim.IDENT_TYPE == 'WHEELS':
                parts.append(f"wheelL_{self.sim.IDENT_WHEEL_L}")
                if hasattr(self.sim, 'RAMP_INCLINATION'):
                    parts.append(f"ramp_{self.sim.RAMP_INCLINATION}")

        bag_name = "_".join(parts) + ".bag"
        return bag_name