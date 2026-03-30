# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi
"""

import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 1000, suppress = True)
from base_controllers.utils.common_functions import plotFrameLinear, plotFrame,  plotJoint, sendStaticTransform, launchFileGeneric
from base_controllers.utils.ros_publish import RosPub
from  base_controllers.tracked_robot.utils import maxxi_constants as constants
import base_controllers.params as conf
from numpy import nan
import rospkg
from  base_controllers.tracked_robot.environment.trajectory import Trajectory
from termcolor import colored
import numpy as np
import pinocchio as pin
from base_controllers.open_loop_simulation3d import  TrackedVehicleSimulator3D, Ground3D
from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager
from base_controllers.closed_loop_simulation_chen import GenericSimulator
from base_controllers.utils.common_functions import SafeTFBroadcaster, checkRosMaster
from matplotlib import pyplot as plt
robotName = "tractor" # needs to inherit BaseController

class EvaluateEnergyConsumption(GenericSimulator):
    def __init__(self, dt=None):
        super().__init__(robotName)
        self.initializeEnergyComputation(dt)
        pass


    def mapFromWorldFrameToBaseFrame(self, v_des, omega_des, euler):
        #new way
        w_R_b = self.math_utils.eul2Rot(euler)
        hf_R_b = self.math_utils.eul2Rot(np.array([euler[0], euler[1], 0.]))
        # project v_des which is in Horizontal frame onto hf_x_b
        v_cos_angle = hf_R_b[0].dot(np.array([1., 0., 0.]))
        b_v_des_x = v_des / v_cos_angle
        omega_cos_angle = w_R_b[2].dot(np.array([0., 0., 1.]))
        b_omega_des = omega_des / omega_cos_angle

        #old way
        # w_R_b = self.math_utils.eul2Rot(euler)
        # hf_R_b = self.math_utils.eul2Rot(np.array([euler[0],euler[1], 0.]))
        # # project v_des which is in Horizontal frame onto hf_x_b
        # b_v_des_x = hf_R_b[0].dot(np.array([v_des, 0., 0.]))
        # # project omega_des which is in WF  onto w_z_b
        # b_omega_des = w_R_b[2].dot(np.array([0., 0.,omega_des]))

        return b_v_des_x, b_omega_des

    def  mapToWheels(self, v_des,omega_des):
        # assumes v_des,omega_des in base_frame
        qd_des = np.zeros(2)
        qd_des[0] = (v_des - omega_des * constants.TRACK_WIDTH / 2)/constants.SPROCKET_RADIUS  # left front
        qd_des[1] = (v_des + omega_des * constants.TRACK_WIDTH / 2)/constants.SPROCKET_RADIUS  # right front
        return qd_des

    def initVars(self):
        self.basePoseW = np.zeros(6)
        self.quaternion = np.array([0., 0., 0., 1.])  # fundamental otherwise receivepose gets stuck
        self.q_des = conf.robot_params[self.robot_name]['q_0']
        self.qd_des = np.zeros(2)
        self.b_R_w = np.eye(3)
        self.time = np.zeros(1)
        self.log_counter = 0
        self.des_x = 0.
        self.des_y = 0.
        self.des_yaw = 0.
        self.beta_l= 0.
        self.beta_r= 0.
        self.alpha= 0.

        # log vars
        self.basePoseW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.baseTwistW_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.q_des_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.q_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.qd_des_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.qd_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.time_log = np.full((conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.state_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.des_state_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.basePoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.beta_l_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.beta_r_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.alpha_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.des_x_vec = np.empty(1)
        self.des_y_vec = np.empty(1)
        self.des_yaw_vec = np.empty(1)

    def plotData(self):
        # xy plot
        plt.figure()
        plt.plot(self.des_state_log[0, :], self.des_state_log[1, :], "-ro", label="desired")
        plt.plot(self.state_log[0, :], self.state_log[1, :], "-bo", label="real")
        plt.legend()
        plt.title(f"XY plot: {self.ControlType}, Long: {self.LONG_SLIP_COMPENSATION} Side: {self.SIDE_SLIP_COMPENSATION}")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.figure()
        plt.plot(self.des_x_vec, self.des_y_vec, "-ro", label="planned_low_discr", markersize=10, alpha=0.5)
        valid = np.isfinite(self.des_state_log[0, :])
        plt.plot(self.des_state_log[0, :], self.des_state_log[1, :], "-bo", label="interpolated")
        plt.legend()
        plt.title(f"CHOMP reference: XY plot:")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def logData(self):
        if (self.log_counter<conf.robot_params[self.robot_name]['buffer_size'] ):
            self.des_state_log[0, self.log_counter] = self.des_x
            self.des_state_log[1, self.log_counter] = self.des_y
            self.des_state_log[2, self.log_counter] = self.des_yaw
            self.state_log[0, self.log_counter] = self.basePoseW[self.u.sp_crd["LX"]]
            self.state_log[1, self.log_counter] = self.basePoseW[self.u.sp_crd["LY"]]
            self.state_log[2, self.log_counter] =  self.basePoseW[self.u.sp_crd["AZ"]]
            self.basePoseW_des_log[:, self.log_counter] = self.basePoseW_des #basepose is logged in base controller

            self.alpha_log[self.log_counter] = self.alpha
            self.beta_l_log[self.log_counter] = self.beta_l
            self.beta_r_log[self.log_counter] = self.beta_r

            self.basePoseW_log[:, self.log_counter] = self.basePoseW
            self.baseTwistW_log[:, self.log_counter] = self.baseTwistW
            self.q_des_log[:, self.log_counter] = self.q_des
            self.qd_des_log[:, self.log_counter] = self.qd_des
            self.time_log[self.log_counter] = self.time
            self.log_counter += 1

    def initializeSimulation(self):
        self.terrain_consistent_pose_init = np.array([self.p0[0], self.p0[1], 0, 0, 0, 0]).copy()
        start_position, start_roll, start_pitch = self.terrainManager.project_on_mesh(point=self.terrain_consistent_pose_init[:2], direction=np.array([0., 0., 1.]), base_yaw=self.p0[2])
        self.terrain_consistent_pose_init[:3] = start_position.copy()
        self.terrain_consistent_pose_init[3] = start_roll
        self.terrain_consistent_pose_init[4] = start_pitch
        self.terrain_consistent_pose_init[5] = self.p0[2]

        # init com self.vehicle_param.height above ground
        w_R_terr = self.math_utils.eul2Rot(self.terrain_consistent_pose_init[3:])
        self.terrain_consistent_pose_init[:3] += self.tracked_vehicle_simulator.consider_robot_height * (w_R_terr[:, 2] * self.tracked_vehicle_simulator.vehicle_param.height)

        self.tracked_vehicle_simulator.initSimulation(pose_init=self.terrain_consistent_pose_init, twist_init=np.zeros(6), ros_pub=self.ros_pub if hasattr(self, "ros_pub") else None)
        # important, you need to reset also baseState otherwise robot_state the first time will be set to 0,0,0!
        self.basePoseW = np.copy(self.terrain_consistent_pose_init)
        self.baseTwistW = np.zeros(6)


    def computeCost(self, des_x_vec, des_y_vec, des_yaw_vec, v_ol, omega_ol, plan_dt):
        # compute COST (all the next code should be REAPEATED for any cost computation TODO)
        self.initVars()

        self.des_x_vec = des_x_vec
        self.des_y_vec = des_y_vec
        self.des_yaw_vec = des_yaw_vec
        # IMPORTANT need to override p0[2] pf[2] because I cannot optimize for orientation
        self.p0[2] = self.des_yaw_vec[0]
        self.pf[2] = self.des_yaw_vec[-1]

        if self.DEBUG:
            self.plotChompTraj(self.des_x_vec, self.des_y_vec)
            self.ros_pub.publishVisual(delete_markers=False)

        # set the traj class with chomp output, the traj class with just enable to interpolate chomp samples on a filer grid
        self.traj = Trajectory(None, des_x_vec, des_y_vec, des_yaw_vec, None, DT=plan_dt, v=v_ol, omega=omega_ol)
        traj_length = len(v_ol)
        self.traj.set_initial_time(start_time=self.time)
        # initialize basePose and sim states to p0 configuration
        self.initializeSimulation()
        # initialize traj cost
        Cost = [ ]
        Energy_last = 0
        Energy = 0


        B = constants.TRACK_WIDTH

        while not ros.is_shutdown():

            # get single sample point interpolated from traj with dt = 0.001
            self.des_x, self.des_y, self.des_yaw, self.v_d, self.omega_d, self.v_dot_d, self.omega_dot_d, traj_finished = self.traj.evalTraj(self.time)


            # need to map self.v_d, self.omega_d that are in XY plane into the moving  base frame (Frenet frame)
            self.b_v_d, self.b_omega_d = self.mapFromWorldFrameToBaseFrame(
                self.v_d, self.omega_d, self.basePoseW[3:]
            )

            # map long vel and omega into wheel speeds
            self.qd_des = self.mapToWheels(self.b_v_d, self.b_omega_d)
            # compute wheel positions (for debug)
            self.q_des = self.q_des + self.qd_des * self.dt

            # update  kinematic variables with forward simulation step
            pg, terrain_roll, terrain_pitch = self.terrainManager.project_on_mesh(
                point=self.basePoseW[:2],
                direction=np.array([0., 0., 1.]),
                base_yaw=self.basePoseW[5]
            )

            # terrain yaw is determined by the robot orientation!
            terrain_yaw = self.basePoseW[5]
            pose_des, terrain_roll_des, terrain_pitch_des = self.terrainManager.project_on_mesh(
                point=np.array([self.des_x, self.des_y]),
                direction=np.array([0., 0., 1.]),
                base_yaw=self.des_yaw
            )

            # optional compute normal just for debug
            w_R_terr = self.math_utils.eul2Rot(np.array([terrain_roll, terrain_pitch, terrain_yaw]))
            w_normal = w_R_terr.dot(np.array([0, 0, 1]))

            # pg is the point on ground correspondent to pcom_on_track_level
            self.tracked_vehicle_simulator.simulateOneStep(
                pg, terrain_roll, terrain_pitch, terrain_yaw, self.qd_des[0], self.qd_des[1]
            )
            self.basePoseW, self.baseTwistW = self.tracked_vehicle_simulator.getRobotState()


            # update quaternions
            self.euler = self.u.angPart(self.basePoseW)
            self.quaternion = pin.Quaternion(pin.rpy.rpyToMatrix(self.euler))
            # self.b_R_w = self.math_utils.eul2Rot(self.euler).T
            # shift up  of robot height along Zb component
            pose_des += self.tracked_vehicle_simulator.consider_robot_height * self.tracked_vehicle_simulator.w_com_height_vector
            self.basePoseW_des = np.concatenate((pose_des, np.array([terrain_roll_des, terrain_pitch_des, self.des_yaw])))

            if self.DEBUG:
                self.now = ros.Time.from_sec(self.time)
                # self.clock_pub.publish(Clock(clock=self.now))
                self.broadcaster.sendTransform(self.u.linPart(self.basePoseW), self.quaternion, self.now, '/base_link', '/world')

            # retrieve terra-mechanics interactions from last dynamics update
            Fx_l, Fx_r, Fy_l, Fy_r = self.tracked_vehicle_simulator.getTerramechanicsInteractions()

            # estimate slippages using long vel and omega (in base frame) and  wheel speeds
            self.beta_l, self.beta_r, self.alpha, _, b_vel_xy = self.estimateSlippages(
                self.baseTwistW,
                self.basePoseW[self.u.sp_crd["AZ"]],
                self.qd_des
            )

            # update energy consumption for this sample of the trajectory
            b_v_x = b_vel_xy[0]
            b_v_y = b_vel_xy[1]

            # body angular velocity around z in base frame
            w_R_b = self.math_utils.eul2Rot(self.u.angPart(self.basePoseW))
            b_ang_vel = w_R_b.T.dot(self.u.angPart(self.baseTwistW))
            omega_b = b_ang_vel[2]

            # longitudinal velocity of each track center
            v_track_l = b_v_x - omega_b * B / 2.0
            v_track_r = b_v_x + omega_b * B / 2.0

            # power-like cost:
            #   Fx * v_x     -> longitudinal traction effort
            #   Fx * beta    -> longitudinal slip loss
            #   Fy * v_y     -> lateral slip loss
            P_long_left = Fx_l * (v_track_l + self.beta_l)
            P_long_right = Fx_r * (v_track_r + self.beta_r)
            P_lat_left = Fy_l * b_v_y
            P_lat_right = Fy_r * b_v_y

            P_total = P_long_left + P_long_right + P_lat_left + P_lat_right

            Energy+= P_total * self.dt
            # integrate power over time to get an energy-like quantity
            if self.traj.get_knot_transition(): #accumulate power every plan_dt seconds
                Cost.append(Energy-Energy_last)
                Energy_last = Energy

            if traj_finished and self.DEBUG:
                print(f"Energy:{Energy} =  Cost  {sum(Cost)}, len of cost vector: {len(Cost)}")
                break

            if np.mod(self.time, 1) == 0:
                print(colored(f"TIME: {self.time}", "red"))

            if self.DEBUG:
                self.logData()

            # advance simulation time
            self.time = np.round(self.time + np.array([self.dt]), 4)



        if self.DEBUG:
            self.plotData()

        return np.array(Cost)

    def initializeEnergyComputation(self, dt=None):

        if dt is None:
            self.dt = conf.robot_params[self.robot_name]['dt']
        else:
            self.dt = dt
        #prologue (do only once)
        self.DEBUG = True

        self.friction_coefficient = 0.8  # 0.1 (used only in 2d) / 0.4 (2d and 3d) (used for planning in paper)/ 0.6 (only 3d)  with slopes we need high friction otherwise alpha is too high
        # initial pose
        self.p0 = np.array([0., 0., 0.])
        #final pose
        self.pf = np.array([220 * 0.02, 190 * 0.02,   np.pi / 4])  # 0.02 is the conversion gain to convert units used in chomp_no_theta into meters
        self.PLANNING_DURATION = 20.
        #ovverride default buffer size
        conf.robot_params[self.robot_name]['buffer_size'] = int(self.PLANNING_DURATION  / self.dt)
        self.TERRAIN_TYPE = 'terrain_chen2'  # 'terrain', 'sphere3', 'terrain_chen2'
        self.OBSTACLES = True

        #full detail model
        print(colored("SIMULATION 3D is unstable for dt > 0.001, resetting dt=0.001 and increased 5x buffer_size", "red"))
        groundParams = Ground3D(friction_coefficient=self.friction_coefficient, terrain_stiffness=1e05, terrain_damping=0.5e04)
        self.tracked_vehicle_simulator = TrackedVehicleSimulator3D(dt=self.dt,  ground=groundParams, USE_MESH=self.TERRAIN, enable_visuals=False, contact_distribution=False)
        self.flag3D='_3d_'

        #set terrain
        self.terrainManager = TerrainManager(rospkg.RosPack().get_path('tractor_description') + "/meshes/" + self.TERRAIN_TYPE + ".stl")
        self.tracked_vehicle_simulator.setTerrainManager(self.terrainManager)

        if self.DEBUG:
            import os
            os.system("killall rosmaster rviz gzserver")
            checkRosMaster()
            self.ros_pub = RosPub(self.robot_name, only_visual=True)
            self.broadcaster = SafeTFBroadcaster()
            launchFileGeneric(rospkg.RosPack().get_path('tractor_description') + "/launch/rviz_nojoints.launch")
            # publish ramp in rviz (only for debug)
            # self.ros_pub.add_plane(pos=np.array([0, 0, -0.]), orient=np.array([0., self.RAMP_INCLINATION, 0]),color="white", alpha=0.5)
            # publish mesh in rviz (only for debug)
            self.ros_pub.add_mesh("tractor_description", "/meshes/" + self.TERRAIN_TYPE + ".stl", position=np.array([0., 0., 0.0]), color="red", alpha=1.0)
            if self.OBSTACLES:
                self.ros_pub.add_mesh("tractor_description", '/meshes/obstacles.stl', position=np.array([0., 0., 0.0]),  color="blue", alpha=1.0)
            ros.sleep(1.)
        #finish of the plologue


if __name__ == '__main__':
    p = EvaluateEnergyConsumption()
    #unit test: compute cost with one trajectory
    # plan trajectory with chomp (discretized with plant_dt = 1s)
    des_x_vec, des_y_vec, des_yaw_vec, v_ol, omega_ol, plan_dt = p.getChomp(p.p0, p.pf)
    p.computeCost(des_x_vec, des_y_vec, des_yaw_vec, v_ol, omega_ol, plan_dt)



