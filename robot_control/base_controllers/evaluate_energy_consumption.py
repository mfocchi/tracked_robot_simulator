# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi
"""

from __future__ import print_function
import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 1000, suppress = True)
from base_controllers.utils.common_functions import plotFrameLinear, plotFrame,  plotJoint, sendStaticTransform, launchFileGeneric
from base_controllers.utils.ros_publish import RosPub
from  base_controllers.tracked_robot.utils import maxxi_constants as constants
import params as conf
from numpy import nan
import rospkg
from  base_controllers.tracked_robot.environment.trajectory import Trajectory
from termcolor import colored
import numpy as np
import pinocchio as pin
from base_controllers.open_loop_simulation3d import  TrackedVehicleSimulator3D, Ground3D
from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager
from closed_loop_simulation_chen import GenericSimulator
from base_controllers.utils.common_functions import SafeTFBroadcaster, checkRosMaster
from matplotlib import pyplot as plt
robotName = "tractor" # needs to inherit BaseController


def mapFromWorldFrameToBaseFrame(v_des, omega_des, euler):
    w_R_b = p.math_utils.eul2Rot(euler)
    hf_R_b = p.math_utils.eul2Rot(np.array([euler[0],euler[1], 0.]))
    # project v_des which is in Horizontal frame onto hf_x_b
    b_v_des = hf_R_b[0].dot(np.array([v_des, 0., 0.]))
    # project omega_des which is in WF  onto w_z_b
    b_omega_des = w_R_b[2].dot(np.array([0., 0.,omega_des]))
    return b_v_des, b_omega_des

def   mapToWheels(v_des,omega_des):
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
    self.des_theta = 0.
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
    self.des_theta_vec = np.empty(1)

def plotData(self):
    # xy plot
    plt.figure()
    plt.plot(p.des_state_log[0, :], p.des_state_log[1, :], "-ro", label="desired")
    plt.plot(p.state_log[0, :], p.state_log[1, :], "-bo", label="real")
    plt.legend()
    plt.title(f"XY plot: {p.ControlType}, Long: {p.LONG_SLIP_COMPENSATION} Side: {p.SIDE_SLIP_COMPENSATION}")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)

    plt.figure()
    plt.plot(p.des_x_vec, p.des_y_vec, "-ro", label="planned_low_discr", markersize=10, alpha=0.5)
    valid = np.isfinite(p.des_state_log[0, :])
    plt.plot(p.des_state_log[0, :], p.des_state_log[1, :], "-bo", label="interpolated")
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
        self.des_state_log[2, self.log_counter] = self.des_theta
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
    start_position, start_roll, start_pitch = p.terrainManager.project_on_mesh(point=self.terrain_consistent_pose_init[:2], direction=np.array([0., 0., 1.]), base_yaw=self.p0[2])
    self.terrain_consistent_pose_init[:3] = start_position.copy()
    self.terrain_consistent_pose_init[3] = start_roll
    self.terrain_consistent_pose_init[4] = start_pitch
    self.terrain_consistent_pose_init[5] = self.p0[2]

    # init com self.vehicle_param.height above ground
    w_R_terr = p.math_utils.eul2Rot(self.terrain_consistent_pose_init[3:])
    self.terrain_consistent_pose_init[:3] += self.tracked_vehicle_simulator.consider_robot_height * (w_R_terr[:, 2] * self.tracked_vehicle_simulator.vehicle_param.height)

    self.tracked_vehicle_simulator.initSimulation(pose_init=self.terrain_consistent_pose_init, twist_init=np.zeros(6), ros_pub=self.ros_pub if hasattr(self, "ros_pub") else None)
    # important, you need to reset also baseState otherwise robot_state the first time will be set to 0,0,0!
    self.basePoseW = np.copy(self.terrain_consistent_pose_init)
    self.baseTwistW = np.zeros(6)

if __name__ == '__main__':
    #prologue (do only once)
    p = GenericSimulator(robotName)
    p.DEBUG = True

    p.friction_coefficient = 0.6  # 0.1 (used only in 2d) / 0.4 (2d and 3d) (used for planning in paper)/ 0.6 (only 3d)  with slopes we need high friction otherwise alpha is too high
    # initial pose
    p.p0 = np.array([0., 0., 0.])
    #final pose
    p.pf = np.array([220 * 0.02, 190 * 0.02,   np.pi / 4])  # 0.02 is the conversion gain to convert units used in chomp_no_theta into meters
    p.PLANNING_DURATION = 20.
    #ovverride default buffer size
    conf.robot_params[p.robot_name]['buffer_size'] = int(p.PLANNING_DURATION  / conf.robot_params[p.robot_name]['dt'])
    p.TERRAIN_TYPE = 'terrain'  # 'terrain', 'sphere3'
    p.OBSTACLES = True

    #full detail model
    print(colored("SIMULATION 3D is unstable for dt > 0.001, resetting dt=0.001 and increased 5x buffer_size", "red"))
    groundParams = Ground3D(friction_coefficient=p.friction_coefficient, terrain_stiffness=1e05, terrain_damping=0.5e04)
    p.tracked_vehicle_simulator = TrackedVehicleSimulator3D(dt=conf.robot_params[p.robot_name]['dt'],  ground=groundParams, USE_MESH=p.TERRAIN, enable_visuals=False, contact_distribution=False)
    p.flag3D='_3d_'

    #set terrain
    p.terrainManager = TerrainManager(rospkg.RosPack().get_path('tractor_description') + "/meshes/" + p.TERRAIN_TYPE + ".stl")
    p.tracked_vehicle_simulator.setTerrainManager(p.terrainManager)

    if p.DEBUG:
        import os
        os.system("killall rosmaster rviz gzserver")
        checkRosMaster()
        p.ros_pub = RosPub(p.robot_name, only_visual=True)
        p.broadcaster = SafeTFBroadcaster()
        launchFileGeneric(rospkg.RosPack().get_path('tractor_description') + "/launch/rviz_nojoints.launch")
        # publish ramp in rviz (only for debug)
        # self.ros_pub.add_plane(pos=np.array([0, 0, -0.]), orient=np.array([0., self.RAMP_INCLINATION, 0]),color="white", alpha=0.5)
        # publish mesh in rviz (only for debug)
        p.ros_pub.add_mesh("tractor_description", "/meshes/" + p.TERRAIN_TYPE + ".stl", position=np.array([0., 0., 0.0]), color="red", alpha=1.0)
        if p.OBSTACLES:
            p.ros_pub.add_mesh("tractor_description", '/meshes/obstacles.stl', position=np.array([0., 0., 0.0]),  color="blue", alpha=1.0)
        ros.sleep(1.)
    #finish of the plologue



    #compute COST (all the next code should be REAPEATED for any cost computation TODO)
    initVars(p)
    #plan trajectory with chomp (discretized with plant_dt = 1s)
    p.des_x_vec, p.des_y_vec, p.des_theta_vec, v_ol, omega_ol, p.plan_dt = p.getChomp(p.p0, p.pf)
    # IMPORTANT need to override p0[2] pf[2] because I cannot optimize for orientation
    p.p0[2] = p.des_theta_vec[0]
    p.pf[2] = p.des_theta_vec[-1]

    if p.DEBUG:
        p.plotChompTraj(p.des_x_vec, p.des_y_vec)
        p.ros_pub.publishVisual(delete_markers=False)

    #set the traj class with chomp output, the traj class with just enable to interpolate chomp samples on a filer grid
    p.traj = Trajectory(None, p.des_x_vec, p.des_y_vec, p.des_theta_vec, None, DT=p.plan_dt, v=v_ol, omega=omega_ol)
    traj_length = len(v_ol)
    p.traj.set_initial_time(start_time=p.time)
    #initialize basePose and sim states to p0 configuration
    initializeSimulation(p)
    #initialize traj cost
    C = 0
    while not  ros.is_shutdown():

        #get single sample point interpolated from traj with dt = 0.001
        p.des_x, p.des_y, p.des_theta, p.v_d, p.omega_d, p.v_dot_d, p.omega_dot_d, traj_finished = p.traj.evalTraj(p.time)
        if traj_finished:
            print(f"Energy: {C}")
            break

        # need to map p.v_d, p.omega_d that are in XY plane into the moving  base frame (Frenet frame)
        p.b_v_d, p.b_omega_d = mapFromWorldFrameToBaseFrame(p.v_d, p.omega_d, p.basePoseW[3:])

        #map long vel and omega into wheel speeds
        p.qd_des = mapToWheels(p.b_v_d, p.b_omega_d )
        #compute wheel positions (for debug)
        p.q_des = p.q_des + p.qd_des * conf.robot_params[p.robot_name]['dt']

        # update  kinematic variables with forward simulation step
        pg, terrain_roll, terrain_pitch = p.terrainManager.project_on_mesh(point=p.basePoseW[:2], direction=np.array([0., 0., 1.]), base_yaw=p.basePoseW[5])
        # terrain yaw is determined by the robot orientation!
        terrain_yaw = p.basePoseW[5]
        pose_des, terrain_roll_des, terrain_pitch_des = p.terrainManager.project_on_mesh(point=np.array([p.des_x, p.des_y]), direction=np.array([0., 0., 1.]), base_yaw=p.basePoseW[5])
        w_R_terr = p.math_utils.eul2Rot(np.array([terrain_roll, terrain_pitch, terrain_yaw]))
        w_normal = w_R_terr.dot(np.array([0, 0, 1]))
        # pg is the point on ground correspondent to pcom_on_track_level
        p.tracked_vehicle_simulator.simulateOneStep(pg, terrain_roll, terrain_pitch, terrain_yaw, p.qd_des[0], p.qd_des[1])
        p.basePoseW, p.baseTwistW = p.tracked_vehicle_simulator.getRobotState()
        #update quaternions
        p.euler = p.u.angPart(p.basePoseW)
        p.quaternion = pin.Quaternion(pin.rpy.rpyToMatrix(p.euler))
        p.b_R_w = p.math_utils.eul2Rot(p.euler).T
        # shift up  of robot height along Zb component
        pose_des += p.tracked_vehicle_simulator.consider_robot_height * p.tracked_vehicle_simulator.w_com_height_vector
        p.basePoseW_des = np.concatenate((pose_des, np.array([terrain_roll_des, terrain_pitch_des, p.des_theta])))

        if p.DEBUG:
            p.now = ros.Time.from_sec(p.time)
            #p.clock_pub.publish(Clock(clock=p.now))
            p.broadcaster.sendTransform(p.u.linPart(p.basePoseW), p.quaternion, p.now, '/base_link', '/world')

        #retrieve terra-mechanics interactions from last dynamics update
        Fx_l, Fx_r, Fy_l, Fy_r = p.tracked_vehicle_simulator.getTerramechanicsInteractions()

        #estimate slippages using long vel and omega (in base frame) and  wheel speeds
        p.beta_l, p.beta_r, p.alpha, _, b_vel_xy = p.estimateSlippages(p.baseTwistW, p.basePoseW[p.u.sp_crd["AZ"]], p.qd_des)

        #update energy consumption for this sample of the trajectory
        C += Fx_l * p.beta_l + Fx_r * p.beta_r  +  Fy_l* b_vel_xy[1] + Fy_r* b_vel_xy[1]

        if np.mod(p.time,1) == 0:
            print(colored(f"TIME: {p.time}","red"))
        # log variables
        if p.DEBUG:
            logData(p)
        #update the time (needed for the evaluation of the trajectory that needs to be interpolated with finer grid beetween samples)
        p.time = np.round(p.time + np.array([conf.robot_params[p.robot_name]['dt']]), 4)  # to avoid issues of dt 0.0009999

    if p.DEBUG:
        plotData(p)

