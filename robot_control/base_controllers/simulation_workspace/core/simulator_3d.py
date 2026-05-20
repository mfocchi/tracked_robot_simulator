#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Simulator wrapper for tracked vehicle simulation with terrain.
"""
import numpy as np
import rospy as ros
import rospkg
import pinocchio as pin
from termcolor import colored

from base_controllers.open_loop_simulation3d import TrackedVehicleSimulator3D, Ground3D
from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager
from base_controllers.utils.common_functions import getRobotModelFloating


class Simulator3D:
    """
    Wrapper for 3D tracked vehicle simulation with terrain support.

    Handles initialization, terrain projection, stepping,
    and state retrieval for the 3D physics simulator.
    """

    def __init__(self, sim_parent):
        """
        Args:
            sim_parent: Reference to GenericSimulator instance
        """
        self.sim = sim_parent
        self.dt = getattr(self.sim, "dt", 0.001)
        self.friction_coefficient = self.sim.friction_coefficient
        self.use_terrain = self.sim.TERRAIN
        self.terrain_type = self.sim.TERRAIN_TYPE

        # Will be initialized in setup()
        self.ground = None
        self.tracked_vehicle_simulator = None
        self.terrain_manager = None
        self.robot = None

        # Terrain-consistent initial pose
        self.terrain_consistent_pose_init = None

    def setup(self):
        """Setup the 3D simulation environment with terrain."""
        print(colored(f"Setting up 3D simulator with friction={self.friction_coefficient}, "
                      f"terrain={'enabled' if self.use_terrain else 'disabled'}", "cyan"))

        # Create ground parameters
        self.ground = Ground3D(
            friction_coefficient=self.friction_coefficient,
            terrain_stiffness=1e05,
            terrain_damping=0.5e04
        )

        # Create simulator
        self.tracked_vehicle_simulator = TrackedVehicleSimulator3D(
            dt=self.dt,
            ground=self.ground,
            USE_MESH=self.use_terrain,
            enable_visuals=False,
            contact_distribution=False
        )

        # Load robot model
        self.robot = getRobotModelFloating(self.sim.robot_name)

        # Setup terrain if enabled
        if self.use_terrain:
            self._setup_terrain()

        print(colored("3D simulator initialized", "green"))

    def _setup_terrain(self):
        """Setup terrain mesh and manager."""
        try:
            mesh_path = rospkg.RosPack().get_path('tractor_description') + \
                        f"/meshes/{self.terrain_type}.stl"
        except Exception as exc:
            print(colored(f"Could not resolve terrain package: {exc}", "yellow"))
            self.use_terrain = False
            self.sim.TERRAIN = False
            self.terrain_manager = None
            return

        print(colored(f"Loading terrain mesh: {mesh_path}", "cyan"))

        try:
            self.terrain_manager = TerrainManager(mesh_path)
            self.tracked_vehicle_simulator.setTerrainManager(self.terrain_manager)
        except Exception as exc:
            print(colored(f"Could not load terrain mesh: {exc}", "yellow"))
            self.use_terrain = False
            self.sim.TERRAIN = False
            self.terrain_manager = None
            return

        # Setup obstacles if enabled
        if self.sim.OBSTACLES:
            self._setup_obstacles()

    def _setup_obstacles(self):
        """Load obstacle meshes if available."""
        try:
            obstacle_path = rospkg.RosPack().get_path('tractor_description') + \
                            '/meshes/obstacles.stl'
            # TerrainManager can handle multiple meshes, but for obstacles
            # we typically publish them separately via RViz
            print(colored(f"Obstacles mesh available at: {obstacle_path}", "cyan"))
        except Exception as e:
            print(colored(f"Could not load obstacles: {e}", "yellow"))

    def set_ramp_terrain(self, length=350., width=350., inclination=0.0, origin=None):
        """
        Replace terrain with a simple ramp.

        Args:
            length: Ramp length [m]
            width: Ramp width [m]
            inclination: Ramp inclination [rad]
            origin: Origin point [x, y, z]
        """
        if origin is None:
            origin = np.array([0, 0, 0])

        from base_controllers.tracked_robot.simulator.terrain_manager import create_ramp_mesh

        ramp_mesh = create_ramp_mesh(
            length=length,
            width=width,
            inclination=inclination,
            origin=origin
        )
        self.terrain_manager.set_mesh(ramp_mesh)
        print(colored(f"Terrain set to ramp with inclination={inclination} rad", "cyan"))

    def init_pose(self, p0=None, terrain_consistent_pose_init=None):
        """
        Compute terrain-consistent initial pose.

        Args:
            p0: Initial 2D pose [x, y, theta]
            terrain_consistent_pose_init: Pre-computed or None
        """
        if p0 is None:
            p0 = self.sim.p0

        self.p0 = np.array(p0)

        # Start with flat pose at p0
        self.terrain_consistent_pose_init = np.array([
            self.p0[0], self.p0[1], 0, 0, 0, self.p0[2]
        ])

        if self.use_terrain and self.terrain_manager is not None:
            # Project onto terrain
            start_position, start_roll, start_pitch = self.terrain_manager.project_on_mesh(
                point=self.terrain_consistent_pose_init[:2],
                direction=np.array([0., 0., 1.]),
                base_yaw=self.p0[2]
            )

            self.terrain_consistent_pose_init[:3] = start_position.copy()
            self.terrain_consistent_pose_init[3] = start_roll
            self.terrain_consistent_pose_init[4] = start_pitch
            self.terrain_consistent_pose_init[5] = self.p0[2]

            # Adjust for robot height
            w_R_terr = self.sim.math_utils.eul2Rot(
                self.terrain_consistent_pose_init[3:]
            )
            height_offset = (
                    self.tracked_vehicle_simulator.consider_robot_height *
                    (w_R_terr[:, 2] * self.tracked_vehicle_simulator.vehicle_param.height)
            )
            self.terrain_consistent_pose_init[:3] += height_offset

            print(colored(
                f"Terrain-consistent pose: pos={self.terrain_consistent_pose_init[:3]}, "
                f"rpy={self.terrain_consistent_pose_init[3:]}", "cyan"
            ))
        else:
            # Flat ground
            self.terrain_consistent_pose_init[:3] += (
                    self.tracked_vehicle_simulator.consider_robot_height *
                    np.array([0., 0., 1.]) *
                    self.tracked_vehicle_simulator.vehicle_param.height
            )

        # Set parent's base pose
        self.sim.basePoseW = np.copy(self.terrain_consistent_pose_init)

        print(colored(f"Initial pose set", "cyan"))

    def init_simulation(self):
        """Initialize the simulation with computed pose."""
        self.tracked_vehicle_simulator.initSimulation(
            pose_init=self.terrain_consistent_pose_init,
            twist_init=np.zeros(6),
            ros_pub=self.sim.ros_pub
        )
        print(colored("3D simulation started", "green"))

    def simulate_one_step(self, qd_left, qd_right):
        """
        Advance simulation by one time step with terrain projection.

        Args:
            qd_left: Left wheel velocity [rad/s]
            qd_right: Right wheel velocity [rad/s]

        Returns:
            tuple: (b_eox, b_eoy) tracking errors in base frame
        """
        if self.use_terrain and self.terrain_manager is not None:
            # Project current position onto terrain
            pg, terrain_roll, terrain_pitch = self.terrain_manager.project_on_mesh(
                point=self.sim.basePoseW[:2],
                direction=np.array([0., 0., 1.]),
                base_yaw=self.sim.basePoseW[5]
            )

            # Project desired position onto terrain
            pose_des, terrain_roll_des, terrain_pitch_des = self.terrain_manager.project_on_mesh(
                point=np.array([self.sim.des_x, self.sim.des_y]),
                direction=np.array([0., 0., 1.]),
                base_yaw=self.sim.basePoseW[5]
            )

            terrain_yaw = self.sim.basePoseW[5]

            # Run simulation step
            b_eox, b_eoy = self.tracked_vehicle_simulator.simulateOneStep(
                pg, terrain_roll, terrain_pitch, terrain_yaw,
                qd_left, qd_right
            )

            # Update parent's state
            self.sim.basePoseW, self.sim.baseTwistW = \
                self.tracked_vehicle_simulator.getRobotState()

            # Compute desired pose with height offset
            pose_des += (
                    self.tracked_vehicle_simulator.consider_robot_height *
                    self.tracked_vehicle_simulator.w_com_height_vector
            )

            self.sim.basePoseW_des = np.concatenate([
                pose_des,
                np.array([terrain_roll_des, terrain_pitch_des, self.sim.des_theta])
            ])

            return b_eox, b_eoy
        else:
            # Flat terrain case
            terrain_roll = 0.0
            terrain_pitch = 0.0
            terrain_yaw = 0.0
            pg = np.array([self.sim.basePoseW[0], self.sim.basePoseW[1], 0.0])
            pose_des = np.array([self.sim.des_x, self.sim.des_y, pg[2]])

            b_eox, b_eoy = self.tracked_vehicle_simulator.simulateOneStep(
                pg, terrain_roll, terrain_pitch, terrain_yaw,
                qd_left, qd_right
            )

            self.sim.basePoseW, self.sim.baseTwistW = \
                self.tracked_vehicle_simulator.getRobotState()

            pose_des += (
                    self.tracked_vehicle_simulator.consider_robot_height *
                    self.tracked_vehicle_simulator.w_com_height_vector
            )

            self.sim.basePoseW_des = np.concatenate([
                pose_des,
                np.array([terrain_roll, terrain_pitch, self.sim.des_theta])
            ])

            return b_eox, b_eoy

    def get_robot_state(self):
        """Get current robot state as RobotState dataclass."""
        from .state import RobotState

        state = RobotState()
        state.x = self.sim.basePoseW[self.sim.u.sp_crd["LX"]]
        state.y = self.sim.basePoseW[self.sim.u.sp_crd["LY"]]
        state.z = self.sim.basePoseW[self.sim.u.sp_crd["LZ"]]
        state.roll = self.sim.basePoseW[self.sim.u.sp_crd["AX"]]
        state.pitch = self.sim.basePoseW[self.sim.u.sp_crd["AY"]]
        state.theta = self.sim.basePoseW[self.sim.u.sp_crd["AZ"]]

        state.vx_b = self.sim.baseTwistW[self.sim.u.sp_crd["LX"]]
        state.vy_b = self.sim.baseTwistW[self.sim.u.sp_crd["LY"]]
        state.omega = self.sim.baseTwistW[self.sim.u.sp_crd["AZ"]]

        return state

    def project_on_terrain(self, point_2d, yaw=0.0):
        """
        Project a 2D point onto the terrain.

        Args:
            point_2d: [x, y] position
            yaw: Yaw angle for projection

        Returns:
            tuple: (position_3d, roll, pitch)
        """
        if self.terrain_manager is not None:
            return self.terrain_manager.project_on_mesh(
                point=point_2d,
                direction=np.array([0., 0., 1.]),
                base_yaw=yaw
            )
        else:
            return (
                np.array([point_2d[0], point_2d[1], 0.0]),
                0.0,
                0.0
            )

    def publish_terrain_visuals(self, ros_pub):
        """
        Publish terrain mesh and obstacles to RViz.

        Args:
            ros_pub: ROS publisher helper instance
        """
        if self.use_terrain:
            ros_pub.add_mesh(
                "tractor_description",
                f"/meshes/{self.terrain_type}.stl",
                position=np.array([0., 0., 0.0]),
                color="red",
                alpha=1.0
            )

            if self.sim.OBSTACLES:
                try:
                    ros_pub.add_mesh(
                        "tractor_description",
                        '/meshes/obstacles.stl',
                        position=np.array([0., 0., 0.0]),
                        color="blue",
                        alpha=1.0
                    )
                except Exception as e:
                    print(colored(f"Could not publish obstacles: {e}", "yellow"))

    @property
    def vehicle_height(self):
        """Get vehicle height parameter."""
        return self.tracked_vehicle_simulator.vehicle_param.height

    def cleanup(self):
        """Cleanup simulator resources."""
        self.tracked_vehicle_simulator = None
        self.ground = None
        self.terrain_manager = None
        print(colored("3D simulator cleaned up", "yellow"))
