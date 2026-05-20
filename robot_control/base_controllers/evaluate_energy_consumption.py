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
from scipy.sparse.linalg import spsolve

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

    def computeTerrainHeightGrid(self, nx=150, ny=150, samples_per_cell=1, z_margin=5.0):
        """
        Discretize the terrain mesh into a regular XY grid and compute
        the terrain height z for each cell.

        nx, ny:
            Number of cells in x and y direction.

        samples_per_cell:
            1 means use only the center of each cell.
            3 means sample 3x3 points inside each cell and average them.

        z_margin:
            How far above the terrain the ray starts.

        Returns:
            X, Y, Z, x_edges, y_edges

            X, Y, Z are arrays of shape (ny, nx).
            Z[i, j] is the average terrain height at that cell.
        """

        import open3d as o3d

        # Get terrain bounds
        bbox = self.terrainManager.mesh.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.get_min_bound())
        max_bound = np.asarray(bbox.get_max_bound())

        x_min, y_min, z_min = min_bound
        x_max, y_max, z_max = max_bound

        # Grid edges and centers
        x_edges = np.linspace(x_min, x_max, nx + 1)
        y_edges = np.linspace(y_min, y_max, ny + 1)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        Xc, Yc = np.meshgrid(x_centers, y_centers)

        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]

        # Rays start above the terrain and go downward.
        z_origin = z_max + z_margin
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        origins_list = []
        cell_ids_list = []

        # One sample at the cell center
        if samples_per_cell <= 1:
            origins = np.column_stack((
                Xc.ravel(),
                Yc.ravel(),
                np.full(nx * ny, z_origin)
            ))

            cell_ids = np.arange(nx * ny)

        # Multiple samples per cell, then average
        else:
            offsets = (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 0.5
            x_offsets = offsets * dx
            y_offsets = offsets * dy

            for ox in x_offsets:
                for oy in y_offsets:
                    Xs = Xc + ox
                    Ys = Yc + oy

                    origins_sample = np.column_stack((
                        Xs.ravel(),
                        Ys.ravel(),
                        np.full(nx * ny, z_origin)
                    ))

                    origins_list.append(origins_sample)
                    cell_ids_list.append(np.arange(nx * ny))

            origins = np.vstack(origins_list)
            cell_ids = np.concatenate(cell_ids_list)

        # Build Open3D ray tensor: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
        directions = np.tile(direction, (origins.shape[0], 1))
        rays_np = np.hstack((origins, directions)).astype(np.float32)

        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)

        # Cast rays onto terrain
        ans = self.terrainManager.scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()

        # Compute hit z coordinate
        hit_z = np.full(origins.shape[0], np.nan)

        valid = np.isfinite(t_hit)
        hit_z[valid] = origins[valid, 2] + t_hit[valid] * direction[2]

        # Average samples inside each grid cell
        z_sum = np.zeros(nx * ny)
        z_count = np.zeros(nx * ny)

        valid_z = np.isfinite(hit_z)

        np.add.at(z_sum, cell_ids[valid_z], hit_z[valid_z])
        np.add.at(z_count, cell_ids[valid_z], 1)

        Z_flat = np.full(nx * ny, np.nan)
        valid_cells = z_count > 0
        Z_flat[valid_cells] = z_sum[valid_cells] / z_count[valid_cells]

        Z = Z_flat.reshape(ny, nx)

        return Xc, Yc, Z, x_edges, y_edges

    def saveTerrainHeightGrid(self, X, Y, Z, folder="/root/ros_ws/src/tracked_robot_simulator"):
        """
        Save terrain grid as CSV files.

        Saves:
            terrain_height_points.csv:
                columns: x, y, z

            terrain_height_matrix.csv:
                only the Z matrix
        """

        import os

        points_path = os.path.join(folder, "terrain_height_points.csv")
        matrix_path = os.path.join(folder, "terrain_height_matrix.csv")

        data = np.column_stack((
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ))

        np.savetxt(
            points_path,
            data,
            delimiter=",",
            header="x,y,z",
            comments="",
            fmt="%.6f"
        )

        np.savetxt(
            matrix_path,
            Z,
            delimiter=",",
            fmt="%.6f"
        )

        print(colored(f"Saved terrain point cloud grid to: {points_path}", "green"))
        print(colored(f"Saved terrain height matrix to: {matrix_path}", "green"))

    def plotTerrainHeightGrid(self, X, Y, Z, x_edges, y_edges, save_path=None, show=False):
        """
        Plot the terrain height map.

        If save_path is provided, the figure is saved to disk.
        If show=True, matplotlib tries to open an interactive window.
        """

        if save_path is None:
            save_path = "/root/ros_ws/src/tracked_robot_simulator/terrain_height_map.png"

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal"
        )

        fig.colorbar(im, ax=ax, label="terrain height z [m]")

        # Contour lines help identify hills and valleys
        ax.contour(
            X,
            Y,
            Z,
            levels=20,
            linewidths=0.5
        )

        ax.set_title("Terrain height map")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)

        print(colored(f"Saved terrain height map image to: {save_path}", "green"))

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)

    def printTerrainHeightCandidates(self, X, Y, Z, n=10):
        """
        Print the n lowest and n highest terrain cells.
        Useful for selecting start and goal points.
        """

        valid = np.isfinite(Z)

        if not np.any(valid):
            print(colored("No valid terrain height points found.", "red"))
            return

        indices = np.argwhere(valid)
        z_values = Z[valid]

        order_low = np.argsort(z_values)[:n]
        order_high = np.argsort(z_values)[-n:][::-1]

        print(colored("\nLowest terrain points:", "cyan"))

        for k in order_low:
            iy, ix = indices[k]
            print(
                f"x={X[iy, ix]: .3f}, "
                f"y={Y[iy, ix]: .3f}, "
                f"z={Z[iy, ix]: .3f}, "
                f"cell=({iy}, {ix})"
            )

        print(colored("\nHighest terrain points:", "cyan"))

        for k in order_high:
            iy, ix = indices[k]
            print(
                f"x={X[iy, ix]: .3f}, "
                f"y={Y[iy, ix]: .3f}, "
                f"z={Z[iy, ix]: .3f}, "
                f"cell=({iy}, {ix})"
            )

        print(colored("\nHeight statistics:", "yellow"))
        print(f"min z:  {np.nanmin(Z):.3f}")
        print(f"max z:  {np.nanmax(Z):.3f}")
        print(f"mean z: {np.nanmean(Z):.3f}")
        print(f"std z:  {np.nanstd(Z):.3f}")

    def makePlanarPose(self, start_xy, goal_xy):
        """
        Create a planar pose [x, y, yaw] from start point toward goal point.
        """

        yaw = np.arctan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0]
        )

        return np.array([
            start_xy[0],
            start_xy[1],
            yaw
        ])

    def plotTerrainHeightGridWithPaths(
            self,
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            paths_m=None,
            labels=None,
            save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map_with_paths.png",
            show=False
    ):
        """
        Plot terrain height heatmap and overlay one or more paths.

        Parameters
        ----------
        X, Y, Z:
            Terrain grid returned by computeTerrainHeightGrid().

        x_edges, y_edges:
            Grid edges returned by computeTerrainHeightGrid().

        paths_m:
            List of paths. Each path must be an array of shape (N, 2),
            containing x, y positions in meters.

        labels:
            List of labels, one for each path.

        save_path:
            Output image path.

        show:
            If True, open an interactive matplotlib window.
        """

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal"
        )

        fig.colorbar(im, ax=ax, label="terrain height z [m]")

        ax.contour(
            X,
            Y,
            Z,
            levels=20,
            linewidths=0.5
        )

        if paths_m is not None:
            if labels is None:
                labels = [f"path {i}" for i in range(len(paths_m))]

            for path, label in zip(paths_m, labels):
                path = np.asarray(path)

                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    "-o",
                    linewidth=2,
                    markersize=3,
                    label=label
                )

                ax.plot(
                    path[0, 0],
                    path[0, 1],
                    "go",
                    markersize=8
                )

                ax.plot(
                    path[-1, 0],
                    path[-1, 1],
                    "rx",
                    markersize=8
                )

        ax.set_title("Terrain height map with CHOMP path")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)

        if paths_m is not None:
            ax.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)

        print(colored(f"Saved terrain height map with paths to: {save_path}", "green"))

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)

    def animateChompHistoryOnTerrain(
            self,
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            chomp_history_m,
            save_path="/root/ros_ws/src/tracked_robot_simulator/chomp_iterations.gif",
            interval=250,
            show=False
    ):
        """
        Animate CHOMP path evolution on top of the terrain height map.

        Parameters
        ----------
        X, Y, Z:
            Terrain grid returned by computeTerrainHeightGrid().

        x_edges, y_edges:
            Grid edges returned by computeTerrainHeightGrid().

        chomp_history_m:
            List of arrays. Each array must have shape (N, 2),
            containing the CHOMP path at one iteration in meters.

        save_path:
            Output GIF path.

        interval:
            Delay between frames in milliseconds.

        show:
            If True, open an interactive matplotlib window.
        """

        from matplotlib.animation import FuncAnimation, PillowWriter

        if chomp_history_m is None or len(chomp_history_m) == 0:
            print(colored("No CHOMP history to animate.", "red"))
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal"
        )

        fig.colorbar(im, ax=ax, label="terrain height z [m]")

        ax.contour(
            X,
            Y,
            Z,
            levels=20,
            linewidths=0.5
        )

        path_line, = ax.plot([], [], "-o", linewidth=2, markersize=3)
        start_point, = ax.plot([], [], "go", markersize=8)
        goal_point, = ax.plot([], [], "rx", markersize=8)

        iteration_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7)
        )

        ax.set_title("CHOMP iterations on terrain height map")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)

        def init():
            path_line.set_data([], [])
            start_point.set_data([], [])
            goal_point.set_data([], [])
            iteration_text.set_text("")
            return path_line, start_point, goal_point, iteration_text

        def update(frame_idx):
            path = np.asarray(chomp_history_m[frame_idx])

            path_line.set_data(path[:, 0], path[:, 1])
            start_point.set_data([path[0, 0]], [path[0, 1]])
            goal_point.set_data([path[-1, 0]], [path[-1, 1]])

            iteration_text.set_text(
                f"CHOMP iteration {frame_idx + 1}/{len(chomp_history_m)}"
            )

            return path_line, start_point, goal_point, iteration_text

        anim = FuncAnimation(
            fig,
            update,
            frames=len(chomp_history_m),
            init_func=init,
            interval=interval,
            blit=True
        )

        fps = max(1, int(1000 / interval))

        anim.save(
            save_path,
            writer=PillowWriter(fps=fps)
        )

        print(colored(f"Saved CHOMP iteration animation to: {save_path}", "green"))

        if show:
            plt.show(block=True)
        else:
            plt.close(fig)

    def saveChompHistoryFramesOnTerrain(
            self,
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            chomp_history_m,
            output_folder="/root/ros_ws/src/tracked_robot_simulator/chomp_iteration_frames",
            frame_stride=1
    ):
        """
        Save one PNG image per CHOMP iteration.

        This is useful when GIF generation does not work or when you want
        individual frames for a report or video.
        """

        import os

        if chomp_history_m is None or len(chomp_history_m) == 0:
            print(colored("No CHOMP history to save.", "red"))
            return

        os.makedirs(output_folder, exist_ok=True)

        for frame_idx in range(0, len(chomp_history_m), frame_stride):
            path = np.asarray(chomp_history_m[frame_idx])

            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(
                Z,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                aspect="equal"
            )

            fig.colorbar(im, ax=ax, label="terrain height z [m]")

            ax.contour(
                X,
                Y,
                Z,
                levels=20,
                linewidths=0.5
            )

            ax.plot(
                path[:, 0],
                path[:, 1],
                "-o",
                linewidth=2,
                markersize=3,
                label=f"iteration {frame_idx}"
            )

            ax.plot(path[0, 0], path[0, 1], "go", markersize=8)
            ax.plot(path[-1, 0], path[-1, 1], "rx", markersize=8)

            ax.set_title(f"CHOMP iteration {frame_idx}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

            fig.tight_layout()

            frame_path = os.path.join(
                output_folder,
                f"chomp_iteration_{frame_idx:04d}.png"
            )

            fig.savefig(frame_path, dpi=200)
            plt.close(fig)

        print(colored(f"Saved CHOMP iteration frames to: {output_folder}", "green"))

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
            self.plotChompTraj3DSpheres(
                self.des_x_vec,
                self.des_y_vec,
                self.des_yaw_vec
            )
            self.ros_pub.add_mesh("tractor_description", "/meshes/" + self.TERRAIN_TYPE + ".stl",
                                  position=np.array([0., 0., 0.0]), color="red", alpha=1.0)
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

            P_long_left = Fx_l * self.beta_l
            P_long_right = Fx_r * self.beta_r
            P_lat_left = Fy_l * b_v_y
            P_lat_right = Fy_r * b_v_y

            P_total = P_long_left + P_long_right + P_lat_left + P_lat_right

            Energy += P_total * self.dt

            # integrate power over time to get an energy-like quantity
            if self.traj.get_knot_transition(): #accumulate power every plan_dt seconds
                Cost.append(Energy-Energy_last)
                Energy_last = Energy

            if traj_finished:
                if self.DEBUG:
                    print(f"Energy:{Energy} =  Cost  {sum(Cost)}, len of cost vector: {len(Cost)}")
                break

            # if np.mod(self.time, 1) == 0:
            #     print(colored(f"TIME: {self.time}", "red"))

            if self.DEBUG:
                self.logData()

            # advance simulation time
            self.time = np.round(self.time + np.array([self.dt]), 4)


        # if self.DEBUG:
        #     self.plotData()


        return np.array(Cost)

    def initializeEnergyComputation(self, dt=None):

        if dt is None:
            self.dt = conf.robot_params[self.robot_name]['dt']
        else:
            self.dt = dt
        #prologue (do only once)
        self.DEBUG = True

        self.friction_coefficient = 0.4  # 0.1 (used only in 2d) / 0.4 (2d and 3d) (used for planning in paper)/ 0.6 (only 3d)  with slopes we need high friction otherwise alpha is too high
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


''' TERRAIN HEIGHT MAP MAIN

if __name__ == '__main__':
    
    p = EvaluateEnergyConsumption()

    X, Y, Z, x_edges, y_edges = p.computeTerrainHeightGrid(
        nx=150,
        ny=150,
        samples_per_cell=3
    )

    p.saveTerrainHeightGrid(X, Y, Z)

    p.printTerrainHeightCandidates(
        X,
        Y,
        Z,
        n=10
    )

    p.plotTerrainHeightGrid(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map.png",
        show=False
    )

    import sys
    sys.exit()
'''

'''
if __name__ == '__main__':

    p = EvaluateEnergyConsumption()

    test_cases = [
        (
            "case_1_high_to_low",
            np.array([0.0, 0.0]),
            np.array([10.0, -2.0])
        ),
        (
            "case_2_low_to_high",
            np.array([10.0, -2.0]),
            np.array([0.0, 0.0])
        ),
        (
            "case_3_hill_avoidance",
            np.array([-15.0, -12.0]),
            np.array([-10.0, 5.0])
        ),
    ]

    for case_name, start_xy, goal_xy in test_cases:

        print(colored(f"\nRunning {case_name}", "green"))

        yaw = np.arctan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0]
        )

        p.p0 = np.array([
            start_xy[0],
            start_xy[1],
            yaw
        ])

        p.pf = np.array([
            goal_xy[0],
            goal_xy[1],
            yaw
        ])

        print(colored(f"p0 = {p.p0}", "yellow"))
        print(colored(f"pf = {p.pf}", "yellow"))

        des_x_vec, des_y_vec, des_yaw_vec, v_ol, omega_ol, plan_dt = p.getChomp(
            p.p0,
            p.pf
        )

        cost = p.computeCost(
            des_x_vec,
            des_y_vec,
            des_yaw_vec,
            v_ol,
            omega_ol,
            plan_dt
        )

        print(colored(f"Total energy-like cost for {case_name}: {np.sum(cost)}", "cyan"))
'''

'''
if __name__ == '__main__':

    p = EvaluateEnergyConsumption()

    # --------------------------------------------------
    # 1) Compute terrain height heatmap
    # --------------------------------------------------
    X, Y, Z, x_edges, y_edges = p.computeTerrainHeightGrid(
        nx=150,
        ny=150,
        samples_per_cell=3
    )

    p.saveTerrainHeightGrid(
        X,
        Y,
        Z,
        folder="/root/ros_ws/src/tracked_robot_simulator"
    )

    p.printTerrainHeightCandidates(
        X,
        Y,
        Z,
        n=10
    )

    p.plotTerrainHeightGrid(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map.png",
        show=False
    )

    # --------------------------------------------------
    # 2) Define start and goal
    # --------------------------------------------------
    case_name = "case_1_high_to_low"

    start_xy = np.array([-15.0, -15.0])
    goal_xy = np.array([-10, 5.0])

    yaw = np.arctan2(
        goal_xy[1] - start_xy[1],
        goal_xy[0] - start_xy[0]
    )

    p.p0 = np.array([
        start_xy[0],
        start_xy[1],
        yaw
    ])

    p.pf = np.array([
        goal_xy[0],
        goal_xy[1],
        yaw
    ])

    print(colored(f"\nRunning {case_name}", "green"))
    print(colored(f"p0 = {p.p0}", "yellow"))
    print(colored(f"pf = {p.pf}", "yellow"))

    # --------------------------------------------------
    # 3) Run CHOMP and collect path at each iteration
    # --------------------------------------------------
    (
        des_x_vec,
        des_y_vec,
        des_yaw_vec,
        v_ol,
        omega_ol,
        plan_dt,
        chomp_history_m
    ) = p.getChomp(
        p.p0,
        p.pf,
        return_history=True,
        save_every=1
    )

    print(colored("\nCHOMP output ranges:", "cyan"))
    print(f"des_x min/max = {np.min(des_x_vec):.3f}, {np.max(des_x_vec):.3f}")
    print(f"des_y min/max = {np.min(des_y_vec):.3f}, {np.max(des_y_vec):.3f}")
    print(f"expected start_xy = {start_xy}")
    print(f"expected goal_xy  = {goal_xy}")
    print(f"number of CHOMP saved iterations = {len(chomp_history_m)}")

    # --------------------------------------------------
    # 4) Plot final CHOMP path on terrain heatmap
    # --------------------------------------------------
    final_path = np.column_stack((
        des_x_vec,
        des_y_vec
    ))

    p.plotTerrainHeightGridWithPaths(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        paths_m=[final_path],
        labels=["final CHOMP path"],
        save_path="/root/ros_ws/src/tracked_robot_simulator/terrain_height_map_final_chomp_path.png",
        show=False
    )

    # --------------------------------------------------
    # 5) Save GIF showing CHOMP path evolution
    # --------------------------------------------------
    p.animateChompHistoryOnTerrain(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        chomp_history_m,
        save_path="/root/ros_ws/src/tracked_robot_simulator/chomp_iterations.gif",
        interval=250,
        show=False
    )

    # --------------------------------------------------
    # 6) Optional: save every CHOMP iteration as PNG
    # --------------------------------------------------
    p.saveChompHistoryFramesOnTerrain(
        X,
        Y,
        Z,
        x_edges,
        y_edges,
        chomp_history_m,
        output_folder="/root/ros_ws/src/tracked_robot_simulator/chomp_iteration_frames",
        frame_stride=1
    )

    # --------------------------------------------------
    # 7) Compute energy-like cost along final CHOMP path
    # --------------------------------------------------
    cost = p.computeCost(
        des_x_vec,
        des_y_vec,
        des_yaw_vec,
        v_ol,
        omega_ol,
        plan_dt
    )

    print(colored(f"Total signed energy-like cost for {case_name}: {np.sum(cost)}", "cyan"))'''