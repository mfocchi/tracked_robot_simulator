#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralized plotting functions for simulation data.
"""
import numpy as np
from matplotlib import pyplot as plt
from base_controllers.utils.common_functions import plotFrameLinear, plotFrame
from base_controllers.tracked_robot.utils import maxxi_constants as constants


class Plotter:
    """
    Contains all matplotlib plotting methods that were originally in the
    GenericSimulator class.
    """

    def __init__(self, sim):
        self.sim = sim

    def plot_all(self):
        """Execute all standard plots."""
        if not hasattr(self.sim, 'time_log') or len(self.sim.time_log) == 0:
            print("No data to plot")
            return
        if not np.any(np.isfinite(self.sim.time_log)):
            print("No data to plot")
            return

        self.plot_xy_trajectory()
        self.plot_control_commands()
        if hasattr(self.sim, 'PLANNING') and self.sim.PLANNING == "chomp":
            self.plot_chomp_reference()
        self.plot_wheel_commands()
        self.plot_states()
        self.plot_base_velocities()
        self.plot_slippage_variables()
        if self.sim.ControlType != 'OPEN_LOOP':
            self.plot_tracking_errors()
        plt.show()

    # ---------------------------------------------------------------
    #   Individual plot methods
    # ---------------------------------------------------------------
    def plot_xy_trajectory(self):
        plt.figure()
        plt.plot(self.sim.des_state_log[0, :], self.sim.des_state_log[1, :], "-ro", label="desired")
        plt.plot(self.sim.state_log[0, :], self.sim.state_log[1, :], "-bo", label="real")
        plt.legend()
        plt.title(f"XY plot: {self.sim.ControlType}, Long: {self.sim.LONG_SLIP_COMPENSATION} "
                  f"Side: {self.sim.SIDE_SLIP_COMPENSATION}")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

    def plot_control_commands(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.sim.time_log, self.sim.ctrl_v_log, "-b", label="REAL")
        plt.plot(self.sim.time_log, self.sim.v_d_log, "-r", label="desired")
        plt.legend()
        plt.title("control commands: v and omega")
        plt.ylabel("linear velocity[m/s]")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.sim.time_log, self.sim.ctrl_omega_log, "-b", label="REAL")
        plt.plot(self.sim.time_log, self.sim.omega_d_log, "-r", label="desired")
        plt.legend()
        plt.xlabel("time[sec]")
        plt.ylabel("angular velocity[rad/s]")
        plt.grid(True)

    def plot_chomp_reference(self):
        if not hasattr(self.sim, 'des_x_vec') or self.sim.des_x_vec is None:
            return
        # chomp xy
        plt.figure()
        plt.plot(self.sim.des_x_vec, self.sim.des_y_vec, "-ro", label="planned_low_discr", markersize=10, alpha=0.5)
        valid = np.isfinite(self.sim.des_state_log[0, :])
        plt.plot(self.sim.des_state_log[0, valid], self.sim.des_state_log[1, valid], "-bo", label="interpolated")
        plt.legend()
        plt.title("CHOMP reference: XY plot")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        # chomp theta
        plt.figure()
        plt.plot(self.sim.time_log, self.sim.des_state_log[2, :], "-bo", label="interpolated")
        if hasattr(self.sim, 'des_theta_vec') and self.sim.des_theta_vec is not None:
            plt.plot(range(len(self.sim.des_theta_vec)), self.sim.des_theta_vec, "-ro", label="planned", markersize=10,
                     alpha=0.5)
        plt.legend()
        plt.title("CHOMP reference: theta plot")
        plt.xlabel("time[s]")
        plt.ylabel("theta[rad]")
        plt.grid(True)

    def plot_wheel_commands(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        plt.suptitle("wheel commands")
        axs[0].plot(self.sim.time_log, self.sim.qd_log[0, :], "-b", linewidth=3)
        axs[0].plot(self.sim.time_log, self.sim.qd_des_log[0, :], "-r", linewidth=4)
        axs[0].plot(self.sim.time_log, constants.MAXSPEED_RADS_PULLEY * np.ones(len(self.sim.time_log)), "-k",
                    linewidth=4)
        axs[0].plot(self.sim.time_log, -constants.MAXSPEED_RADS_PULLEY * np.ones(len(self.sim.time_log)), "-k",
                    linewidth=4)
        axs[0].set_ylabel("WHEEL_L")
        axs[0].grid(True)
        axs[1].plot(self.sim.time_log, self.sim.qd_log[1, :], "-b", linewidth=3)
        axs[1].plot(self.sim.time_log, self.sim.qd_des_log[1, :], "-r", linewidth=4)
        axs[1].plot(self.sim.time_log, constants.MAXSPEED_RADS_PULLEY * np.ones(len(self.sim.time_log)), "-k",
                    linewidth=4)
        axs[1].plot(self.sim.time_log, -constants.MAXSPEED_RADS_PULLEY * np.ones(len(self.sim.time_log)), "-k",
                    linewidth=4)
        axs[1].set_ylabel("WHEEL_R")
        axs[1].grid(True)
        axs[2].plot(self.sim.time_log, self.sim.alpha_control_log, "-r", linewidth=4)
        axs[2].set_ylabel("alpha_control")
        axs[2].grid(True)
        plt.xlabel("Time [s]")
        plt.tight_layout()

    def plot_states(self):
        if self.sim.SIMULATOR == 'distributed3d':
            plotFrame('position', time_log=self.sim.time_log,
                      des_Pose_log=self.sim.basePoseW_des_log,
                      Pose_log=self.sim.basePoseW_log, title='states', frame='W')
        else:
            plotFrameLinear(name='position', time_log=self.sim.time_log,
                            des_Pose_log=self.sim.des_state_log,
                            Pose_log=self.sim.state_log,
                            custom_labels=(["X", "Y", "THETA"]))

    def plot_base_velocities(self):
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(self.sim.time_log, self.sim.b_base_vel_log[0, :], "-b", label="vx")
        plt.ylabel("b_vx")
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2, sharex=ax1)
        plt.plot(self.sim.time_log, self.sim.b_base_vel_log[1, :], "-b", label="vy")
        plt.ylabel("b_vy")
        plt.legend()
        plt.grid(True)

    def plot_slippage_variables(self):
        plt.figure()
        ax2 = plt.subplot(3, 1, 1)
        plt.plot(self.sim.time_log, self.sim.beta_l_log, "-b", label="real")
        plt.plot(self.sim.time_log, self.sim.beta_l_control_log, "-r", label="control")
        plt.ylabel("beta_l")
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 2, sharex=ax2)
        plt.plot(self.sim.time_log, self.sim.beta_r_log, "-b", label="real")
        plt.plot(self.sim.time_log, self.sim.beta_r_control_log, "-r", label="control")
        plt.ylabel("beta_r")
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 3, sharex=ax2)
        plt.plot(self.sim.time_log, self.sim.alpha_log, "-b", label="real")
        plt.plot(self.sim.time_log, self.sim.alpha_control_log, "-r", label="control")
        plt.ylabel("alpha")
        plt.grid(True)
        plt.legend()

    def plot_tracking_errors(self):
        controller = getattr(self.sim, 'controller', None)
        if controller is None:
            controller = getattr(self.sim, 'controller_mgr', None)
        if controller is None:
            return

        if hasattr(controller, 'getErrors'):
            log_e_x, log_e_y, log_e_theta = controller.getErrors()
        elif hasattr(controller, 'get_errors'):
            log_e_x, log_e_y, log_e_theta = controller.get_errors()
        else:
            return

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(np.sqrt(np.array(log_e_x) ** 2 + np.array(log_e_y) ** 2), "-b")
        plt.ylabel("exy")
        plt.title("tracking errors")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(log_e_theta, "-b")
        plt.ylabel("eth")
        plt.grid(True)
