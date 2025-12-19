# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: mfocchi
"""

from __future__ import print_function
import rospy as ros
from base_controllers.utils.math_tools import *
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 1000, suppress = True)
from base_controllers.base_controller import BaseController
from base_controllers.utils.common_functions import plotFrameLinear, plotFrame,  plotJoint, sendStaticTransform, launchFileGeneric
import params as conf
import os
import sys
import rospkg
from gazebo_msgs.srv import SetModelConfiguration
from gazebo_msgs.srv import SetModelConfigurationRequest
from numpy import nan
from matplotlib import pyplot as plt
from base_controllers.utils.math_tools import unwrap_angle
from  base_controllers.tracked_robot.utils import maxxi_constants as constants
from base_controllers.tracked_robot.controllers.lyapunov import LyapunovController, LyapunovParams, Robot
from  base_controllers.tracked_robot.environment.trajectory import Trajectory, ModelsList
from base_controllers.tracked_robot.velocity_generator import VelocityGenerator
from termcolor import colored
from base_controllers.utils.rosbag_recorder import RosbagControlledRecorder
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import numpy as np
import pinocchio as pin
import scipy.io.matlab as mio
from base_controllers.open_loop_simulation2d import TrackedVehicleSimulator, Ground
from base_controllers.utils.common_functions import getRobotModelFloating
from base_controllers.utils.common_functions import checkRosMaster

import pandas as pd
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

robotName = "tractor" # needs to inherit BaseController

class GenericSimulator(BaseController):
    
    def __init__(self, robot_name="tractor"):
        super().__init__(robot_name=robot_name, external_conf = conf)
        self.torque_control = False
        print("Initialized tractor controller---------------------------------------------------------------")
        self.TERRAIN = False #True: Slopes False: Flat terrain
        self.ControlType = 'OPEN_LOOP' #'OPEN_LOOP' 'CLOSED_LOOP_UNICYCLE'
        self.friction_coefficient = 0.1 # 0.1 (used only in 2d) / 0.4 (2d and 3d) (used for planning in paper)/ 0.6 (only 3d)  with slopes we need high friction otherwise alpha is too high
        # initial pose
        self.p0 = np.array([0., 0., 0.])
        # target used only for matlab trajectory generation (dubins/optimization) #need to run dubins_optimization/ros/ros_node.m
        self.pf = np.array([2., 2.5, -0.4])
        self.PLANNING = 'none' # 'none', 'dubins' , 'optim', 'clothoids'
        self.SAVE_BAGS = False
        self.use_ground_truth_contacts = False

    def initVars(self):
        super().initVars()
        ## add your variables to initialize here
        self.ctrl_v = 0.
        self.ctrl_omega = 0.0
        self.v_d = 0.
        self.omega_d = 0.
        self.V= 0.
        self.V_dot = 0.

        self.q_des_q0 = np.zeros(self.robot.na)
        self.ctrl_v_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.ctrl_omega_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.v_d_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.omega_d_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.V_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.V_dot_log = np.empty((conf.robot_params[self.robot_name]['buffer_size']))* nan
        self.des_x = 0.
        self.des_y = 0.
        self.des_theta = 0.
        self.beta_l= 0.
        self.beta_r= 0.
        self.alpha= 0.
        self.alpha_control= 0.
        self.radius = 0.
        self.beta_l_control = 0.
        self.beta_r_control = 0.
        self.log_exy = []
        self.log_e_theta = []
        self.euler = np.zeros(3)
        self.basePoseW_des = np.zeros(6) * np.nan
        self.b_base_vel = np.zeros(2)

        self.state_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.des_state_log = np.full((3, conf.robot_params[self.robot_name]['buffer_size']), np.nan)
        self.basePoseW_des_log = np.full((6, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)
        self.b_base_vel_log = np.full((2, conf.robot_params[self.robot_name]['buffer_size']),  np.nan)

        self.beta_l_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.beta_r_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.alpha_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.alpha_control_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.radius_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.beta_l_control_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.beta_r_control_log = np.empty((conf.robot_params[self.robot_name]['buffer_size'])) * nan
        self.pub_counter = 0
        self.out_of_frequency_counter=0

    def reset_joints(self, q0, joint_names = None):
        # create the message
        req_reset_joints = SetModelConfigurationRequest()
        req_reset_joints.model_name = self.robot_name
        req_reset_joints.urdf_param_name = 'robot_description'
        if joint_names == None:
            req_reset_joints.joint_names = self.joint_names
        else:
            req_reset_joints.joint_names = joint_names
        req_reset_joints.joint_positions = q0
        self.reset_joints_client(req_reset_joints)
        print(colored(f"---------Resetting Joints to: "+str(q0), "blue"))

    def setModelState(self, model_name='', position=np.zeros(3), orientation=np.array([0,0,0,1])):
        # create the message
        set_position = SetModelStateRequest()
        # create model state
        model_state = ModelState()
        model_state.model_name = model_name
        model_state.pose.position.x = position[0]
        model_state.pose.position.y = position[1]
        model_state.pose.position.z = position[2]
        model_state.pose.orientation.x = orientation[0]
        model_state.pose.orientation.y = orientation[1]
        model_state.pose.orientation.z = orientation[2]
        model_state.pose.orientation.w = orientation[3]
        set_position.model_state = model_state
        # send request and get response (in this case none)
        self.set_state(set_position)

    def logData(self):
            if (self.log_counter<conf.robot_params[self.robot_name]['buffer_size'] ):
                ## add your logs here
                self.ctrl_v_log[self.log_counter] = self.ctrl_v
                self.ctrl_omega_log[self.log_counter] = self.ctrl_omega
                self.v_d_log[self.log_counter] = self.v_d
                self.omega_d_log[self.log_counter] = self.omega_d
                self.V_log[self.log_counter] = self.V
                self.V_dot_log[self.log_counter] = self.V_dot
                self.des_state_log[0, self.log_counter] = self.des_x
                self.des_state_log[1, self.log_counter] = self.des_y
                self.des_state_log[2, self.log_counter] = self.des_theta
                self.state_log[0, self.log_counter] = self.basePoseW[self.u.sp_crd["LX"]]
                self.state_log[1, self.log_counter] = self.basePoseW[self.u.sp_crd["LY"]]
                self.state_log[2, self.log_counter] =  self.basePoseW[self.u.sp_crd["AZ"]]

                self.basePoseW_des_log[:, self.log_counter] = self.basePoseW_des #basepose is logged in base controller
                self.b_base_vel_log[:, self.log_counter] = self.b_base_vel  # basepose is logged in base controller

                self.alpha_log[self.log_counter] = self.alpha
                self.beta_l_log[self.log_counter] = self.beta_l
                self.beta_r_log[self.log_counter] = self.beta_r

                self.alpha_control_log[self.log_counter] = self.alpha_control
                self.beta_l_control_log[self.log_counter] = self.beta_l_control
                self.beta_r_control_log[self.log_counter] = self.beta_r_control
                self.radius_log[self.log_counter] = self.radius
            super().logData()

    def startSimulator(self):
        self.decimate_publish = 1
        os.system("killall rosmaster rviz gzserver")
        # launch roscore
        checkRosMaster()
        ros.sleep(1.5)

        # run robot state publisher + load robot description + rviz
        launchFileGeneric(rospkg.RosPack().get_path('tractor_description') + "/launch/rviz_nojoints.launch")
        groundParams = Ground(friction_coefficient=self.friction_coefficient)
        self.tracked_vehicle_simulator = TrackedVehicleSimulator(dt=conf.robot_params[p.robot_name]['dt'], ground=groundParams)
        self.robot = getRobotModelFloating(self.robot_name)
        # instantiating additional publishers
        self.joint_pub = ros.Publisher("/" + self.robot_name + "/joint_states", JointState, queue_size=1)
        self.set_state = ros.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def loadModelAndPublishers(self):
        super().loadModelAndPublishers()
        self.reset_joints_client = ros.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.des_vel_pub = ros.Publisher("/des_vel", JointState, queue_size=1, tcp_nodelay=True)
        #self.clock_pub = ros.Publisher('/clock', Clock, queue_size=10)

        if self.SAVE_BAGS:
            bag_name = f"{p.ControlType}.bag"
            self.recorder = RosbagControlledRecorder(bag_name=bag_name)

    # This will be used instead of the basecontroller one, I do it just to check frequency!
    def _receive_jstate(self, msg):
        for msg_idx in range(len(msg.name)):
            for joint_idx in range(len(self.joint_names)):
                if self.joint_names[joint_idx] == msg.name[msg_idx]:
                    self.q[joint_idx] = msg.position[msg_idx]
                    self.qd[joint_idx] = msg.velocity[msg_idx]
                    self.tau[joint_idx] = msg.effort[msg_idx]

    def checkLoopFrequency(self):
        # check frequency of publishing
        if hasattr(self, 'check_time'):
            loop_time = ros.Time.now().to_sec() - self.check_time  # actual publishing time interval
            ros_loop_time = self.slow_down_factor * conf.robot_params[p.robot_name]['dt'] * self.decimate_publish  # ideal publishing time interval
            if loop_time > 1.3 * (ros_loop_time):
                loop_real_freq = 1 / loop_time  # actual publishing frequency
                freq_ros = 1 / ros_loop_time  # ideal publishing frequency
                print(colored(f"freq mismatch beyond 30%: loop is running at {loop_real_freq} Hz while it should run at {freq_ros} Hz, freq error is {(freq_ros - loop_real_freq) / freq_ros * 100} %", "red"))
                self.out_of_frequency_counter += 1
                if self.out_of_frequency_counter > 10:
                    original_slow_down_factor = self.slow_down_factor
                    self.slow_down_factor *= 2
                    self.rate = ros.Rate(1 / (self.slow_down_factor * conf.robot_params[p.robot_name]['dt']))
                    print(colored(f"increasing slow_down_factor from {original_slow_down_factor} to {self.slow_down_factor}", "red"))
                    self.out_of_frequency_counter = 0

        self.check_time = ros.Time.now().to_sec()

    def getClothoids(self, long_vel, dt = 0.001):
        import Clothoids
        curve = Clothoids.ClothoidCurve("curve")
        curve.build_G1(self.p0[0], self.p0[1], self.p0[2],self.pf[0], self.pf[1], self.pf[2])
        self.PLANNING_DURATION = curve.length() / long_vel
        number_of_samples = int(np.floor(self.PLANNING_DURATION / dt))
        values = np.arange(0, curve.length(), curve.length() / number_of_samples, dtype=np.float64)
        print(colored(f"Planning with {self.PLANNING}, pf: {p.pf},  duration {self.PLANNING_DURATION} s", "red"))
        xy = np.zeros((values.size, 2))
        dxdy = np.zeros((values.size, 2))
        theta = np.zeros((values.size))
        dtheta = np.zeros((values.size))
        for i in range(values.size):
            xy[i, :] = curve.eval(values[i])
            theta[i] = curve.theta(values[i])
            dxdy[i, :] = curve.eval_D(values[i])
            dtheta[i] = curve.theta_D(values[i])
        # map to time (only velocities are affected)
        dxdy_t = dxdy * long_vel
        omega_vec = dtheta * long_vel
        long_v_vec = np.ones((values.size))*long_vel
        return xy[:,0] , xy[:,1], theta, long_v_vec, omega_vec , dt


    def deregister_node(self):
        os.system("killall rosmaster gzserver gzclient rviz")
        super().deregister_node()

    def startupProcedure(self):

        self.tracked_vehicle_simulator.initSimulation(pose_init=self.p0, vbody_init=np.array([0, 0, 0.0]))
        # important, you need to reset also baseState otherwise robot_state the first time will be set to 0,0,0!
        self.basePoseW[self.u.sp_crd["LX"]] = self.p0[0]
        self.basePoseW[self.u.sp_crd["LY"]] = self.p0[1]
        self.basePoseW[self.u.sp_crd["LZ"]] = self.tracked_vehicle_simulator.tracked_robot.vehicle_param.height  # fixed height TODO change this when on slopes
        self.basePoseW[self.u.sp_crd["AZ"]] = self.p0[2]

        self.broadcast_world = False
        self.slow_down_factor = 2

        # loop frequency
        self.rate = ros.Rate(1 / (self.slow_down_factor * conf.robot_params[p.robot_name]['dt']))

    def plotData(self):
        if conf.plotting:
            #xy plot
            plt.figure()
            plt.plot(p.des_state_log[0, :], p.des_state_log[1, :], "-r", label="desired")
            plt.plot(p.state_log[0, :], p.state_log[1, :], "-b", label="real")
            plt.legend()
            plt.title(f"Control: {p.ControlType}")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.axis("equal")
            plt.grid(True)

            # # command plot
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(p.time_log, p.ctrl_v_log, "-b", label="REAL")
            plt.plot(p.time_log, p.v_d_log, "-r", label="desired")
            plt.legend()
            plt.title("v and omega")
            plt.ylabel("linear velocity[m/s]")
            plt.grid(True)
            plt.subplot(2, 1, 2)
            plt.plot(p.time_log, p.ctrl_omega_log, "-b", label="REAL")
            plt.plot(p.time_log, p.omega_d_log, "-r", label="desired")
            plt.legend()
            plt.xlabel("time[sec]")
            plt.ylabel("angular velocity[rad/s]")
            plt.grid(True)


            #plotJoint('position', p.time_log, q_log=p.q_log, q_des_log=p.q_des_log, joint_names=p.joint_names)
            #joint velocities with limits
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))  # Create all 3 subplots at once

            axs[0].plot(p.time_log, p.qd_log[0, :], "-b", linewidth=3)
            axs[0].plot(p.time_log, p.qd_des_log[0, :], "-r", linewidth=4)
            axs[0].plot(p.time_log, constants.MAXSPEED_RADS_PULLEY * np.ones(len(p.time_log)), "-k", linewidth=4)
            axs[0].plot(p.time_log, -constants.MAXSPEED_RADS_PULLEY * np.ones(len(p.time_log)), "-k", linewidth=4)
            axs[0].set_ylabel("WHEEL_L")
            axs[0].grid(True)

            axs[1].plot(p.time_log, p.qd_log[1, :], "-b", linewidth=3)
            axs[1].plot(p.time_log, p.qd_des_log[1, :], "-r", linewidth=4)
            axs[1].plot(p.time_log, constants.MAXSPEED_RADS_PULLEY * np.ones(len(p.time_log)), "-k", linewidth=4)
            axs[1].plot(p.time_log, -constants.MAXSPEED_RADS_PULLEY * np.ones(len(p.time_log)), "-k", linewidth=4)
            axs[1].set_ylabel("WHEEL_R")
            axs[1].grid(True)

            axs[2].plot(p.time_log, p.alpha_control_log, "-r", linewidth=4)
            axs[2].set_ylabel("alpha")
            axs[2].grid(True)

            plt.xlabel("Time [s]")
            plt.tight_layout()
            plt.show()

            #states plot
            plotFrameLinear(name='position',time_log=p.time_log,des_Pose_log = p.des_state_log, Pose_log=p.state_log, custom_labels=(["X","Y","THETA"]))

            #plot velocities in the base frame
            plt.figure()
            ax1 = plt.subplot(2, 1, 1)
            plt.plot(self.time_log, self.b_base_vel_log[0, :], "-b", label="vx")
            plt.ylabel("b_vx")
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 1, 2, sharex=ax1)
            plt.plot(self.time_log, self.b_base_vel_log[1, :], "-b", label="vy")
            plt.ylabel("b_vy")
            plt.legend()
            plt.grid(True)


            if p.ControlType != 'OPEN_LOOP':
                # tracking errors
                p.log_e_x, p.log_e_y, p.log_e_theta = p.controller.getErrors()
                plt.figure()
                plt.subplot(2, 1, 1)
                plt.plot(np.sqrt(np.power(self.log_e_x,2) +np.power(self.log_e_y,2)), "-b")
                plt.ylabel("exy")
                plt.title("tracking errors")
                plt.grid(True)
                plt.subplot(2, 1, 2)
                plt.plot(self.log_e_theta, "-b")
                plt.ylabel("eth")
                plt.grid(True)

    def mapFromWheels(self, wheel_l, wheel_r):
        if not np.isscalar(wheel_l):
            v = np.zeros_like(wheel_l)
            omega = np.zeros_like(wheel_l)
            for i in range(len(wheel_l)):
                v[i] = constants.SPROCKET_RADIUS*(wheel_l[i] + wheel_r[i])/2
                omega[i] = constants.SPROCKET_RADIUS/constants.TRACK_WIDTH*(wheel_r[i] -wheel_l[i])
            return v, omega


    def   mapToWheels(self, v_des,omega_des):

        # assumes v_des,omega_des in base_frame
        qd_des = np.zeros(2)
        qd_des[0] = (v_des - omega_des * constants.TRACK_WIDTH / 2)/constants.SPROCKET_RADIUS  # left front
        qd_des[1] = (v_des + omega_des * constants.TRACK_WIDTH / 2)/constants.SPROCKET_RADIUS  # right front

        #publish des commands as well
        msg = JointState()
        msg.name = self.joint_names
        msg.header.stamp = ros.Time.from_sec(self.time)
        msg.velocity = np.array([v_des, omega_des])
        self.des_vel_pub.publish(msg)
        return qd_des

    #unwrap the joints states
    def unwrap(self):
        for i in range(self.robot.na):
            self.q[i], self.q_old[i] =unwrap_angle(self.q[i], self.q_old[i])

    def generateWheelTraj(self, wheel_l = -4.5):
        ####################################
        # OPEN LOOP wl , wr (from -IDENT_MAX_WHEEL_SPEED to IDENT_MAX_WHEEL_SPEED)
        ####################################

        wheel_l_vec = []
        wheel_r_vec = []
        change_interval = 2.
        if wheel_l <= 0.: #this is to make such that the ID starts always with no rotational speed
            wheel_r = np.linspace(-10, 10, 32) #it if passes from 0 for some reason there is a non linear
                #behaviour in the long slippage
        else:
            wheel_r =np.linspace(-10, 10, 32)
        time = 0
        i = 0
        while True:
            time = np.round(time + conf.robot_params[p.robot_name]['dt'], 4)
            wheel_l_vec.append(wheel_l)
            wheel_r_vec.append(wheel_r[i])
            # detect_switch = not(round(math.fmod(time,change_interval),3) >0)
            if time > ((1 + i) * change_interval):
                i += 1
            if i == len(wheel_r):
                break
        wheel_l_vec.append(0.0)
        wheel_r_vec.append(0.0)
        return wheel_l_vec,wheel_r_vec

    def generateOpenLoopTraj(self, R_initial= 0.05, R_final=0.6, increment=0.025, dt = 0.005, long_v = 0.1, direction="left"):
        # only around 0.3
        change_interval = 6.
        increment = increment
        turning_radius_vec = np.arange(R_final, R_initial, -increment)
        if direction=='left':
            ang_w = np.round(long_v / turning_radius_vec, 3)  # [rad/s]
        else:
            ang_w = -np.round(long_v / turning_radius_vec, 3)  # [rad/s]
        omega_vec = []
        v_vec = []
        time = 0
        i = 0
        while True:
            time = np.round(time + dt, 3)
            omega_vec.append(ang_w[i])
            v_vec.append(long_v)
            # detect_switch = not(round(math.fmod(time,change_interval),3) >0)
            if time > ((1 + i) * change_interval):
                i += 1
            if i == len(turning_radius_vec):
                break
        v_vec.append(0.0)
        omega_vec.append(0.0)
        return v_vec, omega_vec

    def estimateSlippages(self,W_baseTwist, theta, qd):
        wheel_L = qd[0]
        wheel_R = qd[1]


        w_vel_xy = np.zeros(2)
        w_vel_xy[0] = W_baseTwist[self.u.sp_crd["LX"]]
        w_vel_xy[1] = W_baseTwist[self.u.sp_crd["LY"]]
        omega = W_baseTwist[self.u.sp_crd["AZ"]]
        # compute BF velocity
        w_R_b = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        b_vel_xy = (w_R_b.T).dot(w_vel_xy)

        b_vel_x = b_vel_xy[0]
        v = np.linalg.norm(b_vel_xy)

        # compute turning radius for logging
        # in the case radius is infinite, betas are zero (this is to avoid Nans)
        if (abs(omega) < 1e-05) and (abs(v) > 1e-05):
            radius = 1e08 * np.sign(v)
        elif (abs(omega) < 1e-05) and (abs(v) < 1e-05):
            radius = 1e8
        else:
            radius = v / (omega)

        # track velocity  from encoder
        v_enc_l = constants.SPROCKET_RADIUS *  wheel_L
        v_enc_r = constants.SPROCKET_RADIUS *  wheel_R
        B = constants.TRACK_WIDTH

        v_track_l = b_vel_x - omega* B / 2
        v_track_r = b_vel_x + omega* B / 2
        
        # discrepancy bw what it turn out to be (real track) and what it
        # should be (desired) from encoder
        beta_l = v_enc_l-v_track_l
        beta_r = v_enc_r-v_track_r  
        if (abs(b_vel_xy[1])<0.00001) or (abs(b_vel_xy[0])<0.00001):
            side_slip = 0.
        else:
            side_slip = math.atan2(b_vel_xy[1],b_vel_xy[0])

        return beta_l, beta_r, side_slip, radius, b_vel_xy

    
    def send_des_jstate(self, q_des, qd_des, tau_ffwd):

        # comment this if it is too slow
        #self.checkLoopFrequency()

        # No need to change the convention because in the HW interface we use our conventtion (see ros_impedance_contoller_xx.yaml)
        msg = JointState()
        msg.name = self.joint_names
        msg.header.stamp = ros.Time.from_sec(self.time)
        msg.position = q_des
        msg.velocity = qd_des
        msg.effort = tau_ffwd
        if np.mod(self.pub_counter, self.decimate_publish) == 0:
            self.pub_des_jstate.publish(msg) #publish in /commands

        # I comment because it slows loop down TODO
        if np.mod(self.pub_counter, self.decimate_publish) == 0:
            self.joint_pub.publish(msg)  # this publishes in tractor/joint_state q = q_des, it is just for rviz to see the joints of the wheels moving

            self.tracked_vehicle_simulator.simulateOneStep(qd_des[0], qd_des[1])
            pose, pose_der =  self.tracked_vehicle_simulator.getRobotState()
            #fill in base state
            self.basePoseW[:2] = pose[:2]
            self.basePoseW[self.u.sp_crd["AZ"]] = pose[2]
            self.baseTwistW[:2] = pose_der[:2]
            self.baseTwistW[self.u.sp_crd["AZ"]] = pose_der[2]

            self.euler = self.u.angPart(self.basePoseW)
            self.quaternion = pin.Quaternion(pin.rpy.rpyToMatrix(self.euler))
            self.b_R_w = self.math_utils.eul2Rot(self.euler).T
            #publish TF for rviz
            self.broadcaster.sendTransform(self.u.linPart(self.basePoseW),
                                           self.quaternion,
                                           ros.Time.from_sec(self.time), '/base_link', '/world')

            self.q = q_des.copy()
            self.qd = qd_des.copy()

        if self.TERRAIN: #this is published to show mesh in rviz
            self.ros_pub.add_mesh("tractor_description", "/meshes/terrain.stl", position=np.array([0., 0., 0.0]), color="red", alpha=1.0)

        if np.mod(self.time,1) == 0:
            print(colored(f"TIME: {self.time}","red"))
        self.pub_counter+=1

    def pub_odom_msg(self, odom_publisher):
        msg = Odometry()
        msg.header.stamp = ros.Time.from_sec(self.time)
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

def talker(p):
    p.start()
    p.startSimulator()
    p.loadModelAndPublishers()
    p.u.putIntoGlobalParamServer("use_sim_time", True)
    p.robot.na = 2 #initialize properly vars for only 2 actuators (other 2 are caster wheels)
    p.initVars()
    p.q_old = np.zeros(2)
    p.initSubscribers()
    p.startupProcedure()
    #init joints
    p.q_des = np.copy(p.q_des_q0)
    p.q_old = np.zeros(2)
    robot_state = Robot()
    ros.sleep(1.)
    #
    p.q_des = np.zeros(2)
    p.qd_des = np.zeros(2)
    p.tau_ffwd = np.zeros(2)


    if p.SAVE_BAGS:
        p.recorder.start_recording_srv()
    # OPEN loop control
    if p.ControlType == 'OPEN_LOOP':
        counter = 0
        # generic open loop test
        v_ol = np.linspace(0.4, 0.4, np.int32(20./conf.robot_params[p.robot_name]['dt']))
        omega_ol = np.linspace(0.2, 0.2, np.int32(20./conf.robot_params[p.robot_name]['dt']))
        traj_length = len(v_ol)

        if p.PLANNING == 'none':
            p.des_x = p.p0[0]  # +0.1
            p.des_y = p.p0[1]  # +0.1
            p.des_theta = p.p0[2]  # +0.1
            p.traj = Trajectory(ModelsList.UNICYCLE, p.des_x, p.des_y, p.des_theta, DT=conf.robot_params[p.robot_name]['dt'], v=v_ol, omega=omega_ol)
        else: #matlab or clothoids
            if p.PLANNING=='clothoids':
                des_x_vec, des_y_vec,des_theta_vec, v_ol, omega_ol, plan_dt= p.getClothoids(long_vel=0.4, dt = 0.001)
            p.traj = Trajectory(None, des_x_vec, des_y_vec,des_theta_vec, None, DT=plan_dt, v=v_ol, omega=omega_ol)
            traj_length = len(v_ol)

        while not ros.is_shutdown():

            _, _, _, p.v_d, p.omega_d, _, _, traj_finished = p.traj.evalTraj(p.time)
            p.qd_des = p.mapToWheels(p.v_d, p.omega_d)
            if traj_finished:
                break
            p.tau_ffwd = np.zeros(p.robot.na)
            p.q_des = p.q_des + p.qd_des * conf.robot_params[p.robot_name]['dt']
            # this is just for logging
            p.des_x, p.des_y, p.des_theta, p.v_d, p.omega_d, p.v_dot_d, p.omega_dot_d, _ = p.traj.evalTraj(p.time)
            #senting it to be tracked from the impedance loop
            p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
            p.ros_pub.publishVisual(delete_markers=False)
            p.beta_l, p.beta_r, p.alpha, p.radius, p.b_base_vel = p.estimateSlippages(p.baseTwistW, p.basePoseW[p.u.sp_crd["AZ"]], p.qd)
            # log variables
            p.logData()
            # wait for synconization of the control loop
            p.rate.sleep()
            p.time = np.round(p.time + np.array([conf.robot_params[p.robot_name]['dt']]),  4)  # to avoid issues of dt 0.0009999
    else:
        # CLOSE loop control
        # generate reference trajectory
        vel_gen = VelocityGenerator(simulation_time=20.,    DT=conf.robot_params[p.robot_name]['dt'])
        if p.PLANNING == 'none':
            p.des_x = p.p0[0]
            p.des_y = p.p0[1]
            p.des_theta = p.p0[2]

            if p.friction_coefficient == 0.1:
                v_ol, omega_ol, v_dot_ol, omega_dot_ol, _ = vel_gen.velocity_mir_smooth(v_max_=0.2, omega_max_=0.3)
            if p.friction_coefficient == 0.4:
                v_ol, omega_ol, v_dot_ol, omega_dot_ol, _ = vel_gen.velocity_mir_smooth(v_max_=0.4, omega_max_=0.2)
            if p.friction_coefficient == 0.6:
                v_ol, omega_ol, v_dot_ol, omega_dot_ol, _ = vel_gen.velocity_mir_smooth(v_max_=0.6, omega_max_=0.4)
            p.traj = Trajectory(ModelsList.UNICYCLE, start_x=p.des_x, start_y=p.des_y, start_theta=p.des_theta, DT=conf.robot_params[p.robot_name]['dt'],
                                v=v_ol, omega=omega_ol, v_dot=v_dot_ol, omega_dot=omega_dot_ol)
        else:
            if p.PLANNING == 'clothoids':
                # from base_controllers.utils.profiler import Profiler
                # profiler = Profiler(function_name=p.getClothoids)
                des_x_vec, des_y_vec, des_theta_vec, v_ol, omega_ol, plan_dt = p.getClothoids(long_vel=0.4, dt = 0.001)
            p.traj = Trajectory(None, des_x_vec, des_y_vec, des_theta_vec, None, DT=plan_dt, v=v_ol, omega=omega_ol)

        # Lyapunov controller parameters
        params = LyapunovParams(K_P=10., K_THETA=1., DT=conf.robot_params[p.robot_name]['dt'])
        p.controller = LyapunovController(params=params, robot_constants=constants)
        p.traj.set_initial_time(start_time=p.time)
        while not ros.is_shutdown():
            # update kinematics
            robot_state.x = p.basePoseW[p.u.sp_crd["LX"]]
            robot_state.y = p.basePoseW[p.u.sp_crd["LY"]]
            robot_state.theta = p.basePoseW[p.u.sp_crd["AZ"]]

            # get reference from trajectory
            p.des_x, p.des_y, p.des_theta, p.v_d, p.omega_d, p.v_dot_d, p.omega_dot_d, traj_finished = p.traj.evalTraj(p.time)
            if traj_finished:
                break

            # controllers
            if p.ControlType=='CLOSED_LOOP_UNICYCLE':
                p.ctrl_v, p.ctrl_omega, p.V, p.V_dot = p.controller.control_unicycle(robot_state, p.time, p.des_x, p.des_y, p.des_theta, p.v_d, p.omega_d, traj_finished)

            p.qd_des = p.mapToWheels(p.ctrl_v, p.ctrl_omega)

            # senting it to be tracked from the impedance loop
            p.q_des = p.q_des + p.qd_des * conf.robot_params[p.robot_name]['dt']
            p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
            p.ros_pub.publishVisual(delete_markers=False)

            p.beta_l, p.beta_r, p.alpha, p.radius, p.b_base_vel = p.estimateSlippages(p.baseTwistW,p.basePoseW[p.u.sp_crd["AZ"]], p.qd)
            # log variables
            p.logData()
            # wait for synconization of the control loop
            p.rate.sleep()
            p.time = np.round(p.time + np.array([conf.robot_params[p.robot_name]['dt']]), 4) # to avoid issues of dt 0.0009999

    # always save csv when you do ident (bag file deprecated)
    if p.SAVE_BAGS:
        not_nans = ~np.isnan(p.time_log)
        data = pd.DataFrame({
            "time": p.time_log[not_nans],
            "wheel_l": p.qd_des_log[0, not_nans],
            "wheel_r": p.qd_des_log[1, not_nans],
            "roll": p.basePoseW_log[3, not_nans],
            "pitch": p.basePoseW_log[4, not_nans],
            "yaw": p.basePoseW_log[5, not_nans],
            "beta_l": p.beta_l_log[not_nans],
            "beta_r": p.beta_r_log[not_nans],
            "alpha": p.alpha_log[not_nans]})
        output_file = os.environ['LOCOSIM_DIR'] + '/robot_control/base_controllers/data.csv'
        output_dir = os.path.dirname(output_file)
        # creates all missing directories in the path (and won’t complain if they already exist).
        os.makedirs(output_dir, exist_ok=True)
        data.to_csv(output_file, index=False)
        print(colored(f"Data saved to {output_file}", "red"))


        p.recorder.stop_recording_srv()
        if p.ControlType !='OPEN_LOOP':
            filename = f'{p.ControlType}_Long_{p.LONG_SLIP_COMPENSATION}_Side_{p.SIDE_SLIP_COMPENSATION}.mat'
            p.log_e_x, p.log_e_y, p.log_e_theta = p.controller.getErrors()
            mio.savemat(filename, {'time': p.time_log, 'des_state': p.des_state_log,
                                   'state': p.state_log,
                                   'pose_des':p.basePoseW_des_log,
                                   'pose':p.basePoseW_log, 'ex': p.log_e_x, 'ey': p.log_e_y, 'etheta': p.log_e_theta,
                                   'v': p.ctrl_v_log, 'vd': p.v_d_log, 'omega': p.ctrl_omega_log, 'omega_d': p.omega_d_log,
                                   'wheel_l': p.qd_log[0, :], 'wheel_r': p.qd_log[1, :], 'beta_l': p.beta_l_log,
                                   'beta_r': p.beta_r_log, 'beta_l_pred': p.beta_l_control_log, 'beta_r_pred': p.beta_r_control_log,
                                   'alpha': p.alpha_log, 'alpha_pred': p.alpha_control_log, 'radius': p.radius_log})

    if  p.PLANNING!='none' and p.ControlType!='OPEN_LOOP':
        p.log_e_x, p.log_e_y, p.log_e_theta = p.controller.getErrors()
        e_xy = np.sqrt(np.power(p.log_e_x, 2) + np.power(p.log_e_y, 2))
        rmse_xy = np.sqrt(np.mean(e_xy ** 2))
        rmse_theta = np.sqrt(np.mean(np.array(p.log_e_theta) ** 2))
        print(colored(f"Target: {p.pf.reshape(1,3)}, e_xy: {rmse_xy} e_theta {rmse_theta}","red"))

if __name__ == '__main__':
    p = GenericSimulator(robotName)
    try:
        talker(p)
    except (ros.ROSInterruptException, ros.service.ServiceException):
        pass
    if p.SAVE_BAGS:
        p.recorder.stop_recording_srv()
    ros.signal_shutdown("killed")
    p.deregister_node()
    print("Plotting")
    p.plotData()


