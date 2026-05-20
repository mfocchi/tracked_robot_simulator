#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main GenericSimulator class that coordinates the split simulation modules.
"""
from __future__ import print_function

import os

import numpy as np
import pinocchio as pin
import rospkg
import rospy as ros
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from termcolor import colored

import base_controllers.params as conf
from base_controllers.base_controller import BaseController
from base_controllers.tracked_robot.utils import maxxi_constants as constants
from base_controllers.utils.common_functions import (
    checkRosMaster,
    getRobotModelFloating,
    launchFileGeneric,
)

from .simulator_2d import Simulator2D
from .simulator_3d import Simulator3D
from .state import RobotState
from ..control.controller_manager import ControllerManager
from ..control.wheel_mapper import WheelMapper
from ..logging_utils.data_logger import DataLogger
from ..planning.trajectory_planner import TrajectoryPlanner
from ..slippage.compensator import SlipCompensator
from ..slippage.estimator import SlippageEstimator
from ..visualization.plotter import Plotter
from ..visualization.rviz_publisher import RvizPublisher

CHOMP_TEST_CASES = {
    "high_to_low": {
        "full_name": "case_1_high_to_low",
        "start_xy": np.array([0.0, 0.0]),
        "goal_xy": np.array([10.0, -2.0]),
    },
    "low_to_high": {
        "full_name": "case_2_low_to_high",
        "start_xy": np.array([10.0, -2.0]),
        "goal_xy": np.array([0.0, 0.0]),
    },
    "hill_avoidance": {
        "full_name": "case_3_hill_avoidance",
        "start_xy": np.array([-15.0, -12.0]),
        "goal_xy": np.array([-10.0, 5.0]),
    },
    "through_two_hills":{
        "full_name": "case_4_through_two_hills",
        "start_xy": np.array([-18.0, -5.0]),
        "goal_xy": np.array([15.0, 0.0])
    }
}

class GenericSimulator(BaseController):
    """
    Main simulator orchestrator for tracked vehicle simulation.

    The original code mixed orchestration, ROS publishing, plotting, planning,
    and physics stepping in one large script. This class restores that flow in a
    modular form while keeping legacy-facing method names so the existing
    entrypoint and helpers still line up.
    """

    def __init__(self, robot_name="tractor"):
        super().__init__(robot_name=robot_name, external_conf=conf)
        print(colored("=" * 80, "green"))
        print(colored("Initialized Generic Tractor Simulator", "green"))
        print(colored("=" * 80, "green"))

        self.robot_name = robot_name
        self._init_configuration()

        self.wheel_mapper = None
        self.estimator = None
        self.compensator = None
        self.controller_mgr = None
        self.controller = None
        self.logger = None
        self.plotter = None
        self.rviz_pub = None
        self.planner = None
        self.env_simulator = None
        self.traj = None

        self._simulator_started = False
        self._publishers_loaded = False
        self._vars_initialized = False
        self._subscribers_initialized = False
        self._planned_trajectory_visualized = False
        self._buffer_rescaled_for_3d = False
        self._warned_missing_ros_pub = False
        self._warned_estimator_failure = False

    def _init_configuration(self):
        """Set configuration defaults used by the modular simulator."""
        robot_params = conf.robot_params.get(self.robot_name, {})

        self.SIMULATOR = "distributed3d"
        self.TERRAIN = True

        self.ControlType = "CLOSED_LOOP_UNICYCLE"
        self.SIDE_SLIP_COMPENSATION = "NONE"
        self.LONG_SLIP_COMPENSATION = "NONE"
        self.SLIPPAGE_INFERENCE_TYPE = "decision_trees"
        self.ESTIMATE_ALPHA_WITH_ACTUAL_VALUES = True

        self.friction_coefficient = 0.6
        self.dt = robot_params.get("dt", 0.001)

        self.TEST_CASE_NAME = "high_to_low"
        self.TEST_CASE_FULL_NAME = None

        self.p0 = np.array([0.0, 0.0, 0.0])
        self.pf = np.array([10.0, -2.0, 0.0])

        self.set_chomp_test_case(self.TEST_CASE_NAME)

        self.PLANNING = "chomp"
        self.PLANNING_DURATION = 20.0
        self.PLANNING_SPEED = 0.4
        self.TERRAIN_TYPE = "terrain_chen2"
        self.CHOMP_COST_NAME = "terrain_geometry"
        self.CHOMP_GRADIENT_NAME = "finite_difference"
        self.CHOMP_N_KNOTS = 40
        self.CHOMP_DT = 1.0
        self.CHOMP_MAX_ITER = 100
        self.CHOMP_TOL = 1.0
        self.CHOMP_ETA = 0.001
        self.CHOMP_LAMBDA_SMOOTH = 200.0
        self.CHOMP_SAVE_HISTORY = True
        self.CHOMP_GRID_NX = 150
        self.CHOMP_GRID_NY = 150
        self.CHOMP_GRID_SAMPLES_PER_CELL = 1
        self.CHOMP_GRID_Z_MARGIN = 5.0

        self.IDENT_TYPE = "NONE"
        self.IDENT_LONG_SPEED = 0.2
        self.IDENT_DIRECTION = "left"
        self.IDENT_MAX_WHEEL_SPEED = 18.0
        self.IDENT_WHEEL_L = 0.0
        self.RAMP_INCLINATION = 0.0

        self.SAVE_BAGS = False
        self.OBSTACLES = False
        self.flag3D = "_3d_" if self.SIMULATOR == "distributed3d" else ""
        self.use_ground_truth_contacts = False

        self.ctrl_v = 0.0
        self.ctrl_omega = 0.0
        self.v_d = 0.0
        self.omega_d = 0.0
        self.v_dot_d = 0.0
        self.omega_dot_d = 0.0
        self.V = 0.0
        self.V_dot = 0.0
        self.des_x = 0.0
        self.des_y = 0.0
        self.des_theta = 0.0
        self.alpha = 0.0
        self.alpha_control = 0.0
        self.beta_l = 0.0
        self.beta_r = 0.0
        self.beta_l_control = 0.0
        self.beta_r_control = 0.0
        self.radius = 0.0
        self.b_base_vel = np.zeros(2)
        self.basePoseW_des = np.zeros(6) * np.nan
        self.euler = np.zeros(3)
        self.terrain_consistent_pose_init = None

        self.pub_counter = 0
        self.out_of_frequency_counter = 0
        self.decimate_publish = 1
        self.broadcast_world = False
        self.slow_down_factor = 2.0
        self.time = 0.0
        self.now = None
        self.b_eox = 0.0
        self.b_eoy = 0.0
        self.des_x_vec = None
        self.des_y_vec = None
        self.des_theta_vec = None
        self.plan_dt = None

        self._set_speed_limits()

    def set_chomp_test_case(self, case_name):
        """
        Select one of the predefined CHOMP test cases.

        Available names:
            - high_to_low
            - low_to_high
            - hill_avoidance

        This function updates:
            self.p0
            self.pf

        using 3D pose format:
            [x, y, theta]
        """

        case_name = str(case_name).strip()

        if case_name not in CHOMP_TEST_CASES:
            available = ", ".join(sorted(CHOMP_TEST_CASES.keys()))
            raise ValueError(
                f"Unknown CHOMP test case '{case_name}'. "
                f"Available cases are: {available}"
            )

        test_case = CHOMP_TEST_CASES[case_name]

        start_xy = np.asarray(test_case["start_xy"], dtype=float)
        goal_xy = np.asarray(test_case["goal_xy"], dtype=float)

        delta = goal_xy - start_xy
        theta0 = np.arctan2(delta[1], delta[0])

        self.p0 = np.array([start_xy[0], start_xy[1], theta0])
        self.pf = np.array([goal_xy[0], goal_xy[1], theta0])

        self.TEST_CASE_NAME = case_name
        self.TEST_CASE_FULL_NAME = test_case["full_name"]

        print(colored("=" * 80, "cyan"))
        print(colored(f"Selected CHOMP test case: {self.TEST_CASE_FULL_NAME}", "cyan"))
        print(colored(f"Start p0: {self.p0}", "cyan"))
        print(colored(f"Goal  pf: {self.pf}", "cyan"))
        print(colored("=" * 80, "cyan"))

    def _set_speed_limits(self):
        """Set speed limits based on the friction coefficient."""
        if self.friction_coefficient == 0.1:
            constants.MAXSPEED_RADS_PULLEY = 10.0
            self.IDENT_MAX_WHEEL_SPEED = 10.0
        elif self.friction_coefficient in [0.4, 0.6]:
            constants.MAXSPEED_RADS_PULLEY = 18.0
            self.IDENT_MAX_WHEEL_SPEED = 18.0

    def initVars(self):
        """Initialize variables after the robot model has been loaded."""
        super().initVars()

        self.dt = conf.robot_params.get(self.robot_name, {}).get("dt", self.dt)

        self.wheel_mapper = WheelMapper(self)
        self.estimator = SlippageEstimator(self)
        self.compensator = SlipCompensator(self)
        self.controller_mgr = ControllerManager(self)
        self.controller = self.controller_mgr
        self.logger = DataLogger(self)
        self.plotter = Plotter(self)
        self.rviz_pub = RvizPublisher(self)
        self.planner = TrajectoryPlanner(self)

        n_actuated = int(getattr(self.robot, "na", 2) or 2)
        self.q_des_q0 = np.zeros(n_actuated)
        self.q_des = np.zeros(n_actuated)
        self.qd_des = np.zeros(2)
        self.tau_ffwd = np.zeros(2)
        self.q_old = np.zeros(2)

        self.logger.init_vars()

        if self.SIDE_SLIP_COMPENSATION != "NONE" or self.LONG_SLIP_COMPENSATION != "NONE":
            self.compensator.init_models()

        self._vars_initialized = True

    def logData(self):
        """Delegate logging to DataLogger while keeping BaseController logs."""
        if self.logger is not None and self.log_counter < conf.robot_params[self.robot_name]["buffer_size"]:
            self.logger.log_data()
        super().logData()

    def startSimulator(self):
        """Set up and launch the requested simulation backend."""
        self.decimate_publish = 1

        if os.name != "nt":
            os.system("killall rosmaster rviz gzserver >/dev/null 2>&1")

        checkRosMaster()
        ros.sleep(1.5)

        self._maybe_launch_rviz()

        if self.SIMULATOR == "distributed3d" and self.dt > 0.001:
            print(colored(
                "3D simulation is unstable for dt > 0.001. Resetting dt to 0.001.",
                "yellow",
            ))
            conf.robot_params[self.robot_name]["dt"] = 0.001
            self.dt = 0.001
            if not self._buffer_rescaled_for_3d:
                conf.robot_params[self.robot_name]["buffer_size"] *= 5
                self._buffer_rescaled_for_3d = True

        if self.SIMULATOR == "distributed3d":
            print(colored("Starting 3D distributed simulator", "cyan"))
            self.env_simulator = Simulator3D(self)
        else:
            print(colored("Starting 2D distributed simulator", "cyan"))
            self.env_simulator = Simulator2D(self)

        self.env_simulator.setup()
        self.robot = getRobotModelFloating(self.robot_name)

        self.joint_pub = ros.Publisher(
            f"/{self.robot_name}/joint_states",
            JointState,
            queue_size=1,
        )

        if self.IDENT_TYPE != "NONE":
            self.PLANNING = "none"
            self.ControlType = "OPEN_LOOP"
            self.groundtruth_pub = ros.Publisher(
                f"/{self.robot_name}/ground_truth",
                Odometry,
                queue_size=1,
                tcp_nodelay=True,
            )

        self._simulator_started = True

    def _maybe_launch_rviz(self):
        """Try to launch RViz, but do not fail the whole run if it is unavailable."""
        try:
            description_path = rospkg.RosPack().get_path("tractor_description")
            launchFileGeneric(description_path + "/launch/rviz_nojoints.launch")
        except Exception as exc:
            print(colored(f"Skipping RViz launch: {exc}", "yellow"))

    def loadModelAndPublishers(self):
        """Load the robot model and create ROS publishers used by the simulator."""
        super().loadModelAndPublishers()

        self.des_vel_pub = ros.Publisher(
            "/des_vel",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.clock_pub = ros.Publisher("/clock", Clock, queue_size=10)

        if (
            self.TERRAIN
            and isinstance(self.env_simulator, Simulator3D)
            and self.env_simulator.terrain_manager is None
        ):
            self.env_simulator._setup_terrain()

            if self.IDENT_TYPE == "WHEELS" and self.env_simulator.terrain_manager is not None:
                from base_controllers.tracked_robot.simulator.terrain_manager import create_ramp_mesh

                ramp_mesh = create_ramp_mesh(
                    length=350.0,
                    width=350.0,
                    inclination=self.RAMP_INCLINATION,
                    origin=np.array([0, 0, 0]),
                )
                self.env_simulator.terrain_manager.set_mesh(ramp_mesh)

        if self.SAVE_BAGS:
            self._setup_bag_recording()

        self._publishers_loaded = True

    def _setup_bag_recording(self):
        """Set up ROS bag recording with an informative filename."""
        from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder

        if self.ControlType == "OPEN_LOOP":
            if self.IDENT_TYPE == "V_OMEGA":
                bag_name = (
                    f"ident_sim_longv_{self.IDENT_LONG_SPEED}_{self.IDENT_DIRECTION}"
                    f"_fr_{self.friction_coefficient}.bag"
                )
            elif self.IDENT_TYPE == "WHEELS":
                if self.SIMULATOR == "distributed3d":
                    bag_name = (
                        f"ident_sim_fr_{self.friction_coefficient}_ramp_"
                        f"{self.RAMP_INCLINATION}_wheelL_{self.IDENT_WHEEL_L}.bag"
                    )
                else:
                    bag_name = (
                        f"ident_sim_fr_{self.friction_coefficient}_wheelL_"
                        f"{self.IDENT_WHEEL_L}.bag"
                    )
            else:
                bag_name = f"open_loop_fr_{self.friction_coefficient}.bag"
        else:
            bag_name = (
                f"{self.ControlType}_Long_{self.LONG_SLIP_COMPENSATION}"
                f"_Side_{self.SIDE_SLIP_COMPENSATION}.bag"
            )

        self.recorder = RosbagControlledRecorder(bag_name=bag_name)

    def _prepare_single_run(self):
        """Initialize state, publishers, and subscribers for a control run."""
        if not self._simulator_started:
            self.startSimulator()
        if not self._publishers_loaded:
            self.loadModelAndPublishers()
        if not self._vars_initialized:
            self.initVars()
        if hasattr(self.robot, "na"):
            self.robot.na = 2
        if not self._subscribers_initialized and hasattr(self, "initSubscribers"):
            self.initSubscribers()
            self._subscribers_initialized = True

        self._set_sim_time_param()
        self._init_command_buffers()
        self.startupProcedure()
        self._start_recording_if_needed()

    def _set_sim_time_param(self):
        if hasattr(self, "u") and hasattr(self.u, "putIntoGlobalParamServer"):
            try:
                self.u.putIntoGlobalParamServer("use_sim_time", True)
            except Exception as exc:
                print(colored(f"Could not set use_sim_time: {exc}", "yellow"))

    def _init_command_buffers(self):
        """Reset command-side state before starting the control loop."""
        self.pub_counter = 0
        self.ctrl_v = 0.0
        self.ctrl_omega = 0.0
        self.V = 0.0
        self.V_dot = 0.0
        self.alpha_control = 0.0
        self.beta_l_control = 0.0
        self.beta_r_control = 0.0
        self.b_eox = 0.0
        self.b_eoy = 0.0
        self.time = 0.0

        q_des_seed = np.copy(getattr(self, "q_des_q0", np.zeros(2)))
        if q_des_seed.shape[0] < 2:
            q_des_seed = np.zeros(2)

        self.q_des = np.array(q_des_seed[:2], dtype=float)
        self.qd_des = np.zeros(2)
        self.tau_ffwd = np.zeros(2)
        self.q_old = np.zeros(2)
        self.now = ros.Time.from_sec(0.0)

    def startupProcedure(self):
        """Initialize the physics simulator and timing state for the run."""
        if self.env_simulator is None:
            raise RuntimeError("Simulation backend is not initialized. Call startSimulator() first.")

        self.env_simulator.init_pose(p0=self.p0)
        self.terrain_consistent_pose_init = getattr(
            self.env_simulator,
            "terrain_consistent_pose_init",
            None,
        )
        self.env_simulator.init_simulation()

        if self.terrain_consistent_pose_init is None:
            self.terrain_consistent_pose_init = np.copy(self.basePoseW)

        self.basePoseW_des = np.copy(self.basePoseW)
        self.broadcast_world = False
        self.slow_down_factor = 2.0
        self.rate = ros.Rate(self._rate_hz())

    def _rate_hz(self):
        dt = max(float(self.dt), 1e-6)
        return 1.0 / (self.slow_down_factor * dt)

    def _start_recording_if_needed(self):
        if self.SAVE_BAGS and hasattr(self, "recorder") and hasattr(self.recorder, "start_recording_srv"):
            try:
                self.recorder.start_recording_srv()
            except Exception as exc:
                print(colored(f"Could not start rosbag recording: {exc}", "yellow"))

    def _prepare_reference_trajectory(self):
        """Create or refresh the reference trajectory for the current mode."""
        self._set_desired_pose_from_start()
        self.traj = self.planner.plan_trajectory(self.PLANNING)
        self._sync_planner_exports()

        wheel_profiles = None
        if self.IDENT_TYPE == "WHEELS":
            wheel_l_vec, wheel_r_vec = self.planner.generate_wheel_traj(self.IDENT_WHEEL_L)
            wheel_profiles = (wheel_l_vec, wheel_r_vec)
            traj_length = len(wheel_l_vec)
            buffer_size = conf.robot_params[self.robot_name]["buffer_size"]
            if traj_length > buffer_size:
                raise RuntimeError(
                    f"Wheel identification trajectory ({traj_length}) exceeds buffer size ({buffer_size})."
                )

        if hasattr(self.traj, "set_initial_time"):
            self.traj.set_initial_time(start_time=self._time_as_float())

        self._planned_trajectory_visualized = False
        self._publish_planned_trajectory_once()
        return wheel_profiles

    def _sync_planner_exports(self):
        """Mirror planner outputs onto the simulator for logging and plotting."""
        self.des_x_vec = getattr(self.planner, "des_x_vec", None)
        self.des_y_vec = getattr(self.planner, "des_y_vec", None)
        self.des_theta_vec = getattr(self.planner, "des_theta_vec", None)
        self.plan_dt = getattr(self.planner, "plan_dt", None)

    def _publish_planned_trajectory_once(self):
        if self._planned_trajectory_visualized:
            return
        if self.rviz_pub is None or self.PLANNING == "none":
            return
        try:
            self.planner.plot_planned_trajectory(self.rviz_pub)
            self._planned_trajectory_visualized = True
        except Exception as exc:
            print(colored(f"Could not publish planned trajectory markers: {exc}", "yellow"))

    def _set_desired_pose_from_start(self):
        """Keep the desired pose consistent with the run's starting pose."""
        if self.terrain_consistent_pose_init is not None and len(self.terrain_consistent_pose_init) >= 6:
            self.des_x = float(self.terrain_consistent_pose_init[0])
            self.des_y = float(self.terrain_consistent_pose_init[1])
            self.des_theta = float(self.terrain_consistent_pose_init[5])
        else:
            self.des_x = float(self.p0[0])
            self.des_y = float(self.p0[1])
            self.des_theta = float(self.p0[2])

    def _eval_trajectory(self):
        """Evaluate the current trajectory at the simulator time."""
        if self.traj is None:
            return self.des_x, self.des_y, self.des_theta, 0.0, 0.0, 0.0, 0.0, True

        values = self.traj.evalTraj(self._time_as_float())
        if len(values) != 8:
            raise RuntimeError(f"Unexpected trajectory output: expected 8 values, got {len(values)}.")
        return values

    def _time_as_float(self):
        return float(np.asarray(getattr(self, "time", 0.0)).reshape(-1)[0])

    def _advance_time(self):
        if hasattr(self, "rate"):
            self.rate.sleep()
        self.time = np.round(self._time_as_float() + float(self.dt), 4)

    def _refresh_ros_time(self):
        self.now = ros.Time.from_sec(self._time_as_float())

    def _current_robot_state(self):
        if self.env_simulator is not None and hasattr(self.env_simulator, "get_robot_state"):
            return self.env_simulator.get_robot_state()

        state = RobotState()
        state.update_from_pose(self.basePoseW, self.baseTwistW)
        return state

    def _estimate_slippages(self):
        if self.estimator is None or not hasattr(self, "qd"):
            return

        try:
            self.beta_l, self.beta_r, self.alpha, self.radius, self.b_base_vel = self.estimator.estimate(
                self.baseTwistW,
                self.basePoseW[self.u.sp_crd["AZ"]],
                self.qd,
            )
        except Exception as exc:
            if not self._warned_estimator_failure:
                print(colored(f"Slippage estimation disabled for this run: {exc}", "yellow"))
                self._warned_estimator_failure = True

    def _publish_visuals(self):
        if self.rviz_pub is None:
            return

        try:
            if self.TERRAIN:
                if self.IDENT_TYPE == "WHEELS" and self.SIMULATOR == "distributed3d":
                    self.rviz_pub.add_plane(
                        pos=np.array([0.0, 0.0, 0.0]),
                        orient=np.array([0.0, self.RAMP_INCLINATION, 0.0]),
                        color="white",
                        alpha=1.0,
                    )
                elif self.env_simulator is not None and hasattr(self.env_simulator, "publish_terrain_visuals"):
                    self.env_simulator.publish_terrain_visuals(self.rviz_pub)

            self.rviz_pub.publish_visuals(delete_markers=False)
        except Exception as exc:
            print(colored(f"Visual publishing skipped: {exc}", "yellow"))

    def _update_pose_cache(self):
        """Update cached pose helpers and publish TF/ground-truth topics."""
        self.euler = self.u.angPart(self.basePoseW)
        self.quaternion = pin.Quaternion(pin.rpy.rpyToMatrix(self.euler))
        self.b_R_w = self.math_utils.eul2Rot(self.euler).T

        if hasattr(self, "broadcaster"):
            self.broadcaster.sendTransform(
                self.u.linPart(self.basePoseW),
                self.quaternion,
                self.now,
                "/base_link",
                "/world",
            )

        if (
            self.IDENT_TYPE != "NONE"
            and hasattr(self, "groundtruth_pub")
            and np.mod(self.pub_counter, self.decimate_publish) == 0
        ):
            self.pub_odom_msg(self.groundtruth_pub)

    def send_des_jstate(self, q_des, qd_des, tau_ffwd):
        """Publish desired joint state and advance the simulation by one step."""
        if hasattr(self, "clock_pub") and self.clock_pub is not None:
            self.clock_pub.publish(Clock(clock=self.now))

        msg = JointState()
        msg.name = list(getattr(self, "joint_names", []))
        msg.header.stamp = self.now
        msg.position = np.asarray(q_des)
        msg.velocity = np.asarray(qd_des)
        msg.effort = np.asarray(tau_ffwd)

        command_pub = getattr(self, "pub_des_jstate", None)
        if command_pub is None:
            command_pub = getattr(self, "des_vel_pub", None)
        if command_pub is None and not self._warned_missing_ros_pub:
            print(colored("No command publisher available; desired joint states will not be published.", "yellow"))
            self._warned_missing_ros_pub = True

        if command_pub is not None and np.mod(self.pub_counter, self.decimate_publish) == 0:
            command_pub.publish(msg)

        if hasattr(self, "joint_pub") and self.joint_pub is not None:
            if np.mod(self.pub_counter, self.decimate_publish) == 0:
                self.joint_pub.publish(msg)

        if (
            self.ControlType != "OPEN_LOOP"
            and self.LONG_SLIP_COMPENSATION != "NONE"
            and (
                np.any(np.asarray(qd_des) > constants.MAXSPEED_RADS_PULLEY)
                or np.any(np.asarray(qd_des) < -constants.MAXSPEED_RADS_PULLEY)
            )
        ):
            print(colored("Wheel speed beyond limits; slip predictions may be unreliable.", "red"))

        if self.env_simulator is not None:
            self.b_eox, self.b_eoy = self.env_simulator.simulate_one_step(qd_des[0], qd_des[1])
            self._update_pose_cache()

        self.q = np.asarray(q_des).copy()
        self.qd = np.asarray(qd_des).copy()
        self._publish_visuals()
        self.pub_counter += 1

    def pub_odom_msg(self, odom_publisher):
        """Publish the current ground-truth odometry message."""
        msg = Odometry()
        msg.header.stamp = self.now
        msg.pose.pose.orientation.x = self.quaternion[0]
        msg.pose.pose.orientation.y = self.quaternion[1]
        msg.pose.pose.orientation.z = self.quaternion[2]
        msg.pose.pose.orientation.w = self.quaternion[3]
        msg.pose.pose.position.x = self.basePoseW[self.u.sp_crd["LX"]]
        msg.pose.pose.position.y = self.basePoseW[self.u.sp_crd["LY"]]
        msg.pose.pose.position.z = self.basePoseW[self.u.sp_crd["LZ"]]
        msg.twist.twist.linear.x = self.baseTwistW[self.u.sp_crd["LX"]]
        msg.twist.twist.linear.y = self.baseTwistW[self.u.sp_crd["LY"]]
        msg.twist.twist.linear.z = self.baseTwistW[self.u.sp_crd["LZ"]]
        msg.twist.twist.angular.x = self.baseTwistW[self.u.sp_crd["AX"]]
        msg.twist.twist.angular.y = self.baseTwistW[self.u.sp_crd["AY"]]
        msg.twist.twist.angular.z = self.baseTwistW[self.u.sp_crd["AZ"]]
        odom_publisher.publish(msg)

    def _receive_jstate(self, msg):
        """Compatibility hook used by the base controller subscriber."""
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.q[joint_idx] = msg.position[msg_idx]
                    self.qd[joint_idx] = msg.velocity[msg_idx]
                    self.tau[joint_idx] = msg.effort[msg_idx]

    def run_open_loop(self):
        """Run the open-loop control flow restored from the legacy script."""
        self._prepare_single_run()
        wheel_profiles = self._prepare_reference_trajectory()

        counter = 0
        wheel_l_vec = wheel_r_vec = None
        if wheel_profiles is not None:
            wheel_l_vec, wheel_r_vec = wheel_profiles

        while not ros.is_shutdown():
            self._refresh_ros_time()

            self.des_x, self.des_y, self.des_theta, self.v_d, self.omega_d, self.v_dot_d, self.omega_dot_d, traj_finished = self._eval_trajectory()

            if wheel_l_vec is not None:
                if counter >= len(wheel_l_vec):
                    print(colored("Open-loop wheel identification accomplished.", "cyan"))
                    break
                self.qd_des = np.array([wheel_l_vec[counter], wheel_r_vec[counter]])
                counter += 1
            else:
                if traj_finished:
                    break
                self.qd_des = self.wheel_mapper.map_to_wheels(self.v_d, self.omega_d)

            self.tau_ffwd = np.zeros(2)
            self.q_des = self.q_des + self.qd_des * self.dt
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            self._estimate_slippages()
            self.logData()
            self._advance_time()

    def run_closed_loop(self):
        """Run the closed-loop controller flow using the modular components."""
        self._prepare_single_run()
        self._prepare_reference_trajectory()

        self.controller_mgr.setup_controller(
            control_type=self.ControlType,
            side_slip_comp=self.SIDE_SLIP_COMPENSATION,
            slippage_inference_type=self.SLIPPAGE_INFERENCE_TYPE,
        )
        self.controller = self.controller_mgr

        while not ros.is_shutdown():
            self._refresh_ros_time()

            robot_state = self._current_robot_state()
            self.des_x, self.des_y, self.des_theta, self.v_d, self.omega_d, self.v_dot_d, self.omega_dot_d, traj_finished = self._eval_trajectory()

            if traj_finished:
                break

            self.ctrl_v, self.ctrl_omega, self.V, self.V_dot, self.alpha_control = self.controller_mgr.compute_control(
                robot_state=robot_state,
                des_x=self.des_x,
                des_y=self.des_y,
                des_theta=self.des_theta,
                v_d=self.v_d,
                omega_d=self.omega_d,
                v_dot_d=self.v_dot_d,
                omega_dot_d=self.omega_dot_d,
                traj_finished=traj_finished,
            )

            self.qd_des = self.wheel_mapper.map_to_wheels(self.ctrl_v, self.ctrl_omega)
            self.beta_l_control = 0.0
            self.beta_r_control = 0.0

            if self.ControlType != "CLOSED_LOOP_UNICYCLE" and self.LONG_SLIP_COMPENSATION != "NONE":
                self.qd_des, self.beta_l_control, self.beta_r_control = self.compensator.compensate(
                    self.ctrl_v,
                    self.ctrl_omega,
                    self.qd_des,
                )

            self.tau_ffwd = np.zeros(2)
            self.q_des = self.q_des + self.qd_des * self.dt
            self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
            self._estimate_slippages()
            self.logData()
            self._advance_time()

    def plotData(self):
        """Delegate plotting to the plotter if plotting is enabled."""
        if not getattr(conf, "plotting", True):
            return
        if self.plotter is None:
            return

        try:
            self.plotter.plot_all()
        except Exception as exc:
            print(colored(f"Plotting skipped: {exc}", "yellow"))
