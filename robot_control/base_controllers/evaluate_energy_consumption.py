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
from  base_controllers.tracked_robot.utils import maxxi_constants as constants
import params as conf
import rospkg
from  base_controllers.tracked_robot.environment.trajectory import Trajectory
from termcolor import colored
import numpy as np
from base_controllers.open_loop_simulation3d import  TrackedVehicleSimulator3D, Ground3D
from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager
from closed_loop_simulation_chen import GenericSimulator
robotName = "tractor" # needs to inherit BaseController


def estimateSlippages(b_lin_vel, b_ang_vel, qd):
    wheel_L = qd[0]
    wheel_R = qd[1]

    b_vel_xy = b_lin_vel[:2]
    omega = b_ang_vel[2]

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
    v_enc_l = constants.SPROCKET_RADIUS * wheel_L
    v_enc_r = constants.SPROCKET_RADIUS * wheel_R
    B = constants.TRACK_WIDTH

    v_track_l = b_vel_x - omega * B / 2
    v_track_r = b_vel_x + omega * B / 2

    # discrepancy bw what it turn out to be (real track) and what it
    # should be (desired) from encoder
    beta_l = v_enc_l - v_track_l
    beta_r = v_enc_r - v_track_r
    if (abs(b_vel_xy[1]) < 0.00001) or (abs(b_vel_xy[0]) < 0.00001):
        side_slip = 0.
    else:
        side_slip = math.atan2(b_vel_xy[1], b_vel_xy[0])

    return beta_l, beta_r, side_slip, radius, b_vel_xy


if __name__ == '__main__':
    p = GenericSimulator(robotName)
    p.initVars()
    p.TERRAIN = True  # True: Slopes False: Flat terrain
    p.friction_coefficient = 0.6  # 0.1 (used only in 2d) / 0.4 (2d and 3d) (used for planning in paper)/ 0.6 (only 3d)  with slopes we need high friction otherwise alpha is too high
    # initial pose
    p.p0 = np.array([0., 0., 0.])
    #final pose
    p.pf = np.array([220 * 0.02, 190 * 0.02,   np.pi / 4])  # 0.02 is the conversion gain to convert units used in chomp_no_theta into meters
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

    # publish ramp in rviz
    # self.ros_pub.add_plane(pos=np.array([0, 0, -0.]), orient=np.array([0., self.RAMP_INCLINATION, 0]),                          color="white", alpha=0.5)

    # publish ramp in rviz
    p.ros_pub.add_mesh("tractor_description", "/meshes/" + p.TERRAIN_TYPE + ".stl",
                       position=np.array([0., 0., 0.0]), color="red", alpha=1.0)
    if p.OBSTACLES:
        p.ros_pub.add_mesh("tractor_description", '/meshes/obstacles.stl', position=np.array([0., 0., 0.0]),
                           color="blue", alpha=1.0)

    #compute COST (repeat for gradient)

    #plan trajectory with chomp (discretized with plant_dt = 1s)
    p.des_x_vec, p.des_y_vec, p.des_theta_vec, v_ol, omega_ol, p.plan_dt = p.getChomp(p.p0, p.pf)
    p.plotChompTraj(p.des_x_vec, p.des_y_vec)
    p.traj = Trajectory(None, p.des_x_vec, p.des_y_vec, p.des_theta_vec, None, DT=p.plan_dt, v=v_ol, omega=omega_ol)
    traj_length = len(v_ol)
    p.traj.set_initial_time(start_time=p.time)

    #compute trajectory cost
    C = 0
    while not  ros.is_shutdown():
        p.now = ros.Time.from_sec(p.time)
        #get single sample point interpolated from traj with dt = 0.001
        p.des_x, p.des_y, p.des_theta, p.v_d, p.omega_d, p.v_dot_d, p.omega_dot_d, traj_finished = p.traj.evalTraj(p.time)
        if traj_finished:
            break

        # need to map p.v_d, p.omega_d that are in XY plane into the Frenet (body) frame
        p.b_v, p.b_omega = mapFromXYToBaseFrame(p.v_d, p.omega_d) #TODO

        p.qd_des = p.mapToWheels(p.b_v, p.b_omega )
        p.q_des = p.q_des + p.qd_des * conf.robot_params[p.robot_name]['dt']

        #compute terrain consistent Z position /orientation
        position, roll, pitch, yaw = p.terrainManager.project_on_mesh(point=np.array([p.des_x, p.des_y]), direction=np.array([0., 0., 1.]))
        p.basePoseW[:3] = position.copy()
        p.basePoseW[3] = roll
        p.basePoseW[4] = pitch
        p.basePoseW[5] = yaw
        # consider com is self.vehicle_param.height above ground
        w_R_terr = p.math_utils.eul2Rot(np.array([roll, pitch, yaw]))
        #overwrite linear part
        p.basePoseW[:3] += p.tracked_vehicle_simulator.consider_robot_height * ( w_R_terr[:, 2] * p.tracked_vehicle_simulator.vehicle_param.height)

        #compute normal and position on mesh
        w_normal = w_R_terr.dot(np.array([0, 0, 1]))

        #compute forward dyn
        Fx_l, Fx_r, Fy_l, Fy_r = p.tracked_vehicle_simulator.computeTerramMechanicsOpenLoop(p.basePoseW[:3], roll, pitch,  yaw, p.qd_des[0], p.qd_des[1])
        p.basePoseW, p.baseTwistW = p.tracked_vehicle_simulator.getRobotState()

        #estimate slippages
        p.beta_l, p.beta_r, p.alpha,_, b_vel_xy = estimateSlippages(p.b_v, p.b_omega, p.qd)

        #p.ros_pub.publishVisual(delete_markers=False)
        # log variables
        p.logData()
        # wait for synconization of the control loop
        p.rate.sleep()
        p.time = np.round(p.time + np.array([conf.robot_params[p.robot_name]['dt']]), 4)  # to avoid issues of dt 0.0009999

        #comoute energy
        C += Fx_l * p.beta_l + Fx_r * p.beta_r  +  Fy_l* b_vel_xy[1] + Fy_r* b_vel_xy[1]

