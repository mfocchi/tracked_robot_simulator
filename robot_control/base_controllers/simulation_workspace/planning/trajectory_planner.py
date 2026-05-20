#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main trajectory planner that orchestrates different planning methods.
"""
import numpy as np
from termcolor import colored

from .clothoid_planner import ClothoidPlanner
from .chomp_planner import ChompPlanner
from .trajectory import Trajectory, ModelsList
from .velocity_generator import VelocityGenerator


class TrajectoryPlanner:
    """
    Orchestrates different path planning methods for the tracked vehicle.

    Supports:
    - Clothoid curves (smooth G1-continuous paths)
    - CHOMP (Covariant Hamiltonian Optimization for Motion Planning)
    - Open-loop trajectories (for system identification)
    - Velocity profile generation
    """

    def __init__(self, sim):
        """
        Args:
            sim: Reference to GenericSimulator instance
        """
        self.sim = sim

        # Planners
        self.clothoid_planner = ClothoidPlanner(sim)
        self.chomp_planner = ChompPlanner(sim)
        self.velocity_generator = VelocityGenerator(
            simulation_time=sim.PLANNING_DURATION,
            DT=sim.dt
        )

        # Stored trajectory data
        self.des_x_vec = None
        self.des_y_vec = None
        self.des_theta_vec = None
        self.v_vec = None
        self.omega_vec = None
        self.plan_dt = None

    def plan_trajectory(self, planning_type=None):
        """
        Generate a trajectory based on the specified planning type.

        Args:
            planning_type: 'clothoids', 'chomp', 'none', or None (uses sim.PLANNING)

        Returns:
            Trajectory object
        """
        if planning_type is None:
            planning_type = self.sim.PLANNING

        if planning_type == 'clothoids':
            return self._plan_clothoid()
        elif planning_type == 'chomp':
            return self._plan_chomp()
        else:
            return self._plan_none()

    def _plan_clothoid(self):
        """Generate a Clothoid curve trajectory."""
        print(colored(f"Planning with Clothoids to target: {self.sim.pf}", "cyan"))

        self.des_x_vec, self.des_y_vec, self.des_theta_vec, \
        self.v_vec, self.omega_vec, self.plan_dt = \
            self.clothoid_planner.get_clothoids(
                start=self.sim.p0,
                goal=self.sim.pf,
                long_vel=self.sim.PLANNING_SPEED,
                dt=self.sim.dt
            )

        return self._create_trajectory()

    def _plan_chomp(self):
        """Generate a CHOMP trajectory."""
        print(colored(f"Planning with CHOMP to target: {self.sim.pf}", "cyan"))

        result = self.chomp_planner.get_chomp(
            start=self.sim.p0,
            goal=self.sim.pf,
            return_history=False
        )

        self.des_x_vec, self.des_y_vec, self.des_theta_vec, \
        self.v_vec, self.omega_vec, self.plan_dt = result

        # Override initial and final orientations (CHOMP doesn't optimize theta)
        self.sim.p0[2] = self.des_theta_vec[0]
        self.sim.pf[2] = self.des_theta_vec[-1]

        return self._create_trajectory()

    def _plan_none(self):
        """Generate a simple unicycle trajectory (no path planning)."""
        if hasattr(self.sim, 'terrain_consistent_pose_init') and \
                self.sim.terrain_consistent_pose_init is not None:
            start_x = self.sim.terrain_consistent_pose_init[0]
            start_y = self.sim.terrain_consistent_pose_init[1]
            start_theta = self.sim.terrain_consistent_pose_init[5]
        else:
            start_x = self.sim.p0[0]
            start_y = self.sim.p0[1]
            start_theta = self.sim.p0[2]

        # Generate velocity profile
        if self.sim.ControlType == 'OPEN_LOOP':
            v_ol, omega_ol = self._generate_open_loop_velocity()
        else:
            v_ol, omega_ol, v_dot_ol, omega_dot_ol = \
                self._generate_closed_loop_velocity()

        # Store for logging_utils
        self.des_x_vec = np.array([start_x])
        self.des_y_vec = np.array([start_y])
        self.des_theta_vec = np.array([start_theta])

        return Trajectory(
            ModelsList.UNICYCLE,
            start_x=start_x,
            start_y=start_y,
            start_theta=start_theta,
            DT=self.sim.dt,
            v=v_ol,
            omega=omega_ol
        )

    def _create_trajectory(self):
        """Create a Trajectory object from planned data."""
        return Trajectory(
            model=None,  # No model for planned trajectories
            start_x=self.des_x_vec,
            start_y=self.des_y_vec,
            start_theta=self.des_theta_vec,
            velocity_generator=None,
            DT=self.plan_dt,
            v=self.v_vec,
            omega=self.omega_vec
        )

    def _generate_open_loop_velocity(self):
        """Generate velocity profile for open-loop control."""
        if self.sim.IDENT_TYPE == 'WHEELS':
            # Wheel identification trajectory
            wheel_l_ol, wheel_r_ol = self.generate_wheel_traj(
                wheel_l=self.sim.IDENT_WHEEL_L
            )
            v_ol, omega_ol = self.sim.wheel_mapper.map_from_wheels(
                wheel_l_ol, wheel_r_ol
            )
            return v_ol, omega_ol
        elif self.sim.IDENT_TYPE == 'V_OMEGA':
            # V/Omega identification trajectory
            v_ol, omega_ol = self.generate_open_loop_traj(
                R_initial=0.1,
                R_final=0.6,
                increment=0.05,
                dt=self.sim.dt,
                long_v=self.sim.IDENT_LONG_SPEED,
                direction=self.sim.IDENT_DIRECTION
            )
            return v_ol, omega_ol
        else:
            # Generic open-loop test
            n_steps = int(20.0 / self.sim.dt)
            v_ol = np.linspace(0.4, 0.4, n_steps)
            omega_ol = np.linspace(0.2, 0.2, n_steps)
            return v_ol, omega_ol

    def _generate_closed_loop_velocity(self):
        """Generate velocity profile for closed-loop control."""
        v_max = self._get_max_velocity()
        omega_max = self._get_max_omega()

        v_ol, omega_ol, v_dot_ol, omega_dot_ol, _ = \
            self.velocity_generator.velocity_mir_smooth(
                v_max_=v_max,
                omega_max_=omega_max
            )

        return v_ol, omega_ol, v_dot_ol, omega_dot_ol

    def _get_max_velocity(self):
        """Get maximum linear velocity based on friction."""
        if self.sim.friction_coefficient == 0.1:
            return 0.2
        elif self.sim.friction_coefficient == 0.4:
            return 0.4
        elif self.sim.friction_coefficient == 0.6:
            return 0.6
        return 0.4

    def _get_max_omega(self):
        """Get maximum angular velocity based on friction."""
        if self.sim.friction_coefficient == 0.1:
            return 0.3
        elif self.sim.friction_coefficient == 0.4:
            return 0.2
        elif self.sim.friction_coefficient == 0.6:
            return 0.4
        return 0.2

    def generate_wheel_traj(self, wheel_l=-4.5):
        """
        Generate wheel velocity trajectories for system identification.

        Args:
            wheel_l: Left wheel velocity to hold constant

        Returns:
            tuple: (wheel_l_vec, wheel_r_vec) arrays of wheel velocities
        """
        dt = self.sim.dt
        max_speed = self.sim.IDENT_MAX_WHEEL_SPEED

        if self.sim.SIMULATOR == 'distributed3d':
            # 3D: shorter trajectory for stability
            number_of_samples = int(np.floor(10.0 / dt))
            wheel_l_vec = np.linspace(wheel_l, wheel_l, 3 * number_of_samples)
            wheel_r_vec = np.linspace(0, max_speed, number_of_samples)
            wheel_r_vec = np.append(
                wheel_r_vec,
                np.linspace(max_speed, -max_speed, 2 * number_of_samples)
            )
        else:
            # 2D: longer trajectory with more resolution
            wheel_l_vec = []
            wheel_r_vec = []
            change_interval = 2.0

            if wheel_l <= 0.0:
                wheel_r = np.linspace(-max_speed, max_speed, 32)
            else:
                wheel_r = np.linspace(max_speed, -max_speed, 32)

            time = 0.0
            i = 0
            while True:
                time = np.round(time + dt, 4)
                wheel_l_vec.append(wheel_l)
                wheel_r_vec.append(wheel_r[i])

                if time > ((1 + i) * change_interval):
                    i += 1
                if i == len(wheel_r):
                    break

            wheel_l_vec.append(0.0)
            wheel_r_vec.append(0.0)

            wheel_l_vec = np.array(wheel_l_vec)
            wheel_r_vec = np.array(wheel_r_vec)

        return wheel_l_vec, wheel_r_vec

    def generate_open_loop_traj(self, R_initial=0.05, R_final=0.6,
                                increment=0.025, dt=0.005, long_v=0.1,
                                direction="left"):
        """
        Generate open-loop trajectory with varying turning radii.

        Args:
            R_initial: Initial turning radius [m]
            R_final: Final turning radius [m]
            increment: Radius increment [m]
            dt: Time step [s]
            long_v: Constant longitudinal velocity [m/s]
            direction: 'left' or 'right'

        Returns:
            tuple: (v_vec, omega_vec) velocity profiles
        """
        change_interval = 6.0
        turning_radius_vec = np.arange(R_final, R_initial, -increment)

        if direction == 'left':
            ang_w = np.round(long_v / turning_radius_vec, 3)
        else:
            ang_w = -np.round(long_v / turning_radius_vec, 3)

        omega_vec = []
        v_vec = []
        time = 0.0
        i = 0

        while True:
            time = np.round(time + dt, 3)
            omega_vec.append(ang_w[i])
            v_vec.append(long_v)

            if time > ((1 + i) * change_interval):
                i += 1
            if i == len(turning_radius_vec):
                break

        v_vec.append(0.0)
        omega_vec.append(0.0)

        return np.array(v_vec), np.array(omega_vec)

    def plot_planned_trajectory(self, ros_pub):
        """
        Visualize the planned trajectory in RViz.

        Args:
            ros_pub: RvizPublisher instance
        """
        if self.des_x_vec is None or self.des_y_vec is None:
            return

        if self.sim.SIMULATOR == 'distributed3d' and self.sim.TERRAIN:
            self._plot_trajectory_3d(ros_pub)
        else:
            self._plot_trajectory_2d(ros_pub)

    def _plot_trajectory_2d(self, ros_pub):
        """Plot trajectory as 2D markers."""
        for blob_x, blob_y in zip(self.des_x_vec, self.des_y_vec):
            ros_pub.add_marker(
                np.array([blob_x, blob_y, self.sim.basePoseW[2]]),
                color="white",
                radius=0.5
            )

    def _plot_trajectory_3d(self, ros_pub):
        """Plot trajectory as 3D spheres projected on terrain."""
        sphere_radius = 0.25
        z_clearance = 0.03

        for blob_x, blob_y, blob_yaw in zip(
                self.des_x_vec, self.des_y_vec, self.des_theta_vec
        ):
            terrain_point, terrain_roll, terrain_pitch = \
                self.sim.env_simulator.terrain_manager.project_on_mesh(
                    point=np.array([blob_x, blob_y]),
                    direction=np.array([0., 0., 1.]),
                    base_yaw=blob_yaw
                )

            marker_position = terrain_point.copy()
            marker_position[2] += sphere_radius + z_clearance

            ros_pub.add_marker(
                marker_position,
                color="white",
                radius=sphere_radius
            )

    def get_trajectory_info(self):
        """Get information about the planned trajectory."""
        if self.des_x_vec is None:
            return "No trajectory planned"

        return {
            'num_points': len(self.des_x_vec),
            'duration': self.sim.PLANNING_DURATION,
            'planning_type': self.sim.PLANNING,
            'start': [self.des_x_vec[0], self.des_y_vec[0], self.des_theta_vec[0]],
            'end': [self.des_x_vec[-1], self.des_y_vec[-1], self.des_theta_vec[-1]]
        }