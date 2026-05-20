#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D Simulator wrapper for tracked vehicle simulation.
"""
import numpy as np
import rospy as ros
from termcolor import colored

from base_controllers.open_loop_simulation2d import TrackedVehicleSimulator, Ground
from base_controllers.utils.common_functions import getRobotModelFloating


class Simulator2D:
    """
    Wrapper for 2D tracked vehicle simulation.

    Handles initialization, stepping, and state retrieval
    for the 2D physics simulator.
    """

    def __init__(self, sim_parent):
        """
        Args:
            sim_parent: Reference to GenericSimulator instance
        """
        self.sim = sim_parent
        self.dt = getattr(self.sim, "dt", 0.001)
        self.friction_coefficient = self.sim.friction_coefficient

        # Will be initialized in setup()
        self.ground = None
        self.tracked_vehicle_simulator = None
        self.robot = None

    def setup(self):
        """Setup the 2D simulation environment."""
        print(colored(f"Setting up 2D simulator with friction={self.friction_coefficient}", "cyan"))

        # Validate friction coefficient
        if self.friction_coefficient not in [0.1, 0.4]:
            print(colored(
                f"Warning: friction coefficient {self.friction_coefficient} "
                f"not standard for 2D simulation (use 0.1 or 0.4)", "yellow"
            ))

        # Create ground parameters
        self.ground = Ground(friction_coefficient=self.friction_coefficient)

        # Create simulator
        self.tracked_vehicle_simulator = TrackedVehicleSimulator(
            dt=self.dt,
            ground=self.ground
        )

        # Load robot model
        self.robot = getRobotModelFloating(self.sim.robot_name)

        print(colored("2D simulator initialized", "green"))

    def init_pose(self, p0=None, terrain_consistent_pose_init=None):
        """
        Set initial pose for the simulation.

        Args:
            p0: Initial 2D pose [x, y, theta]
            terrain_consistent_pose_init: Not used in 2D
        """
        if p0 is None:
            p0 = self.sim.p0

        self.initial_pose = np.array(p0)

        # In 2D, we can directly set the base pose
        self.sim.basePoseW[self.sim.u.sp_crd["LX"]] = self.initial_pose[0]
        self.sim.basePoseW[self.sim.u.sp_crd["LY"]] = self.initial_pose[1]
        self.sim.basePoseW[self.sim.u.sp_crd["LZ"]] = \
            self.tracked_vehicle_simulator.tracked_robot.vehicle_param.height
        self.sim.basePoseW[self.sim.u.sp_crd["AZ"]] = self.initial_pose[2]

        print(colored(f"Initial pose set to: {self.initial_pose}", "cyan"))

    def init_simulation(self):
        """Initialize the simulation with the current pose."""
        self.tracked_vehicle_simulator.initSimulation(
            pose_init=self.initial_pose,
            vbody_init=np.array([0, 0, 0.0])
        )
        print(colored("2D simulation started", "green"))

    def simulate_one_step(self, qd_left, qd_right):
        """
        Advance simulation by one time step.

        Args:
            qd_left: Left wheel velocity [rad/s]
            qd_right: Right wheel velocity [rad/s]

        Returns:
            tuple: (b_eox, b_eoy) tracking errors in base frame
        """
        self.tracked_vehicle_simulator.simulateOneStep(qd_left, qd_right)

        # Get updated state
        pose, pose_der = self.tracked_vehicle_simulator.getRobotState()

        # Update parent's state
        self.sim.basePoseW[:2] = pose[:2]
        self.sim.basePoseW[self.sim.u.sp_crd["AZ"]] = pose[2]
        self.sim.baseTwistW[:2] = pose_der[:2]
        self.sim.baseTwistW[self.sim.u.sp_crd["AZ"]] = pose_der[2]

        # In 2D, tracking errors are zero (no terrain)
        return 0.0, 0.0

    def get_robot_state(self):
        """Get current robot state as RobotState dataclass."""
        from .state import RobotState

        state = RobotState()
        state.x = self.sim.basePoseW[self.sim.u.sp_crd["LX"]]
        state.y = self.sim.basePoseW[self.sim.u.sp_crd["LY"]]
        state.theta = self.sim.basePoseW[self.sim.u.sp_crd["AZ"]]
        state.z = self.sim.basePoseW[self.sim.u.sp_crd["LZ"]]

        state.vx_b = self.sim.baseTwistW[self.sim.u.sp_crd["LX"]]
        state.vy_b = self.sim.baseTwistW[self.sim.u.sp_crd["LY"]]
        state.omega = self.sim.baseTwistW[self.sim.u.sp_crd["AZ"]]

        return state

    @property
    def vehicle_height(self):
        """Get vehicle height parameter."""
        return self.tracked_vehicle_simulator.tracked_robot.vehicle_param.height

    def cleanup(self):
        """Cleanup simulator resources."""
        self.tracked_vehicle_simulator = None
        self.ground = None
        print(colored("2D simulator cleaned up", "yellow"))
