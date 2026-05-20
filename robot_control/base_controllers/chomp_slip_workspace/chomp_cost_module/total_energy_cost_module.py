# base_controllers/chomp_slip_workspace/chomp_cost_module/total_energy_cost_module.py

import numpy as np
import rospy as ros

from base_controllers.tracked_robot.environment.trajectory import Trajectory
from base_controllers.tracked_robot.utils import maxxi_constants as constants

from base_controllers.chomp_slip_workspace.chomp_cost_module.simulation_backed_cost_module import (
    SimulationBackedCostModule,
)


class TotalEnergyCostModule(SimulationBackedCostModule):
    """
    Total terrain-interaction energy cost module.

    Computes:

        P_total = Fx_L * v_track_L
                + Fx_R * v_track_R
                + Fy_L * v_y
                + Fy_R * v_y

    where:

        v_track_L = v_x - omega_z * B / 2
        v_track_R = v_x + omega_z * B / 2

    This is different from SlipEnergyCostModule, which computes:

        P_slip = Fx_L * beta_L
               + Fx_R * beta_R
               + Fy_L * v_y
               + Fy_R * v_y
    """

    name = "total_energy"

    def __init__(self, positive_only=True):
        """
        Parameters
        ----------
        positive_only:
            If True, only positive power is accumulated:

                P = max(0, P)

            This is usually better if you want "consumed" energy rather than
            signed mechanical work.
        """

        super().__init__()
        self.positive_only = positive_only

    def compute_cost_vector(self, trajectory_reference, simulator):
        """
        Compute total terrain-interaction energy along a trajectory.

        Parameters
        ----------
        trajectory_reference:
            Object or dict containing:
                x
                y
                yaw
                v
                omega
                plan_dt or dt

        simulator:
            EvaluateEnergyConsumption-compatible simulator.

        Returns
        -------
        np.ndarray
            Cost vector, one cost value per planning interval.
        """

        (
            des_x_vec,
            des_y_vec,
            des_yaw_vec,
            v_ol,
            omega_ol,
            plan_dt,
        ) = self._unpack_trajectory_reference(trajectory_reference)

        # --------------------------------------------------
        # 1) Reset simulator state
        # --------------------------------------------------
        simulator.initVars()

        simulator.des_x_vec = des_x_vec
        simulator.des_y_vec = des_y_vec
        simulator.des_yaw_vec = des_yaw_vec

        simulator.p0 = np.array(
            [
                des_x_vec[0],
                des_y_vec[0],
                des_yaw_vec[0],
            ],
            dtype=float,
        )

        simulator.pf = np.array(
            [
                des_x_vec[-1],
                des_y_vec[-1],
                des_yaw_vec[-1],
            ],
            dtype=float,
        )

        # --------------------------------------------------
        # 2) Optional debug visualization
        # --------------------------------------------------
        if getattr(simulator, "DEBUG", False):
            if hasattr(simulator, "plotChompTraj3DSpheres"):
                simulator.plotChompTraj3DSpheres(
                    simulator.des_x_vec,
                    simulator.des_y_vec,
                    simulator.des_yaw_vec,
                )

            if hasattr(simulator, "ros_pub") and hasattr(simulator, "TERRAIN_TYPE"):
                simulator.ros_pub.add_mesh(
                    "tractor_description",
                    "/meshes/" + simulator.TERRAIN_TYPE + ".stl",
                    position=np.array([0.0, 0.0, 0.0]),
                    color="red",
                    alpha=1.0,
                )
                simulator.ros_pub.publishVisual(delete_markers=False)

        # --------------------------------------------------
        # 3) Build interpolated trajectory object
        # --------------------------------------------------
        simulator.traj = Trajectory(
            None,
            des_x_vec,
            des_y_vec,
            des_yaw_vec,
            None,
            DT=plan_dt,
            v=v_ol,
            omega=omega_ol,
        )

        simulator.traj.set_initial_time(
            start_time=simulator.time,
        )

        # --------------------------------------------------
        # 4) Initialize physical simulation
        # --------------------------------------------------
        simulator.initializeSimulation()

        # --------------------------------------------------
        # 5) Integrate energy over fine simulation time
        # --------------------------------------------------
        cost_vector = []

        energy = 0.0
        energy_last = 0.0

        B = constants.TRACK_WIDTH

        while not ros.is_shutdown():

            # ----------------------------------------------
            # 5.1) Evaluate desired trajectory
            # ----------------------------------------------
            (
                simulator.des_x,
                simulator.des_y,
                simulator.des_yaw,
                simulator.v_d,
                simulator.omega_d,
                simulator.v_dot_d,
                simulator.omega_dot_d,
                traj_finished,
            ) = simulator.traj.evalTraj(simulator.time)

            # ----------------------------------------------
            # 5.2) Desired velocity in world/horizontal frame
            #      -> desired velocity in robot base frame
            # ----------------------------------------------
            simulator.b_v_d, simulator.b_omega_d = (
                simulator.mapFromWorldFrameToBaseFrame(
                    simulator.v_d,
                    simulator.omega_d,
                    simulator.basePoseW[3:],
                )
            )

            # ----------------------------------------------
            # 5.3) Desired base velocity -> sprocket speeds
            # ----------------------------------------------
            simulator.qd_des = simulator.mapToWheels(
                simulator.b_v_d,
                simulator.b_omega_d,
            )

            simulator.q_des = simulator.q_des + simulator.qd_des * simulator.dt

            # ----------------------------------------------
            # 5.4) Project current robot pose onto terrain
            # ----------------------------------------------
            pg, terrain_roll, terrain_pitch = simulator.terrainManager.project_on_mesh(
                point=simulator.basePoseW[:2],
                direction=np.array([0.0, 0.0, 1.0]),
                base_yaw=simulator.basePoseW[5],
            )

            terrain_yaw = simulator.basePoseW[5]

            # Desired pose projected on terrain, used for debug/logging_utils.
            pose_des, terrain_roll_des, terrain_pitch_des = (
                simulator.terrainManager.project_on_mesh(
                    point=np.array([simulator.des_x, simulator.des_y]),
                    direction=np.array([0.0, 0.0, 1.0]),
                    base_yaw=simulator.des_yaw,
                )
            )

            # ----------------------------------------------
            # 5.5) Simulate one step
            # ----------------------------------------------
            simulator.tracked_vehicle_simulator.simulateOneStep(
                pg,
                terrain_roll,
                terrain_pitch,
                terrain_yaw,
                simulator.qd_des[0],
                simulator.qd_des[1],
            )

            simulator.basePoseW, simulator.baseTwistW = (
                simulator.tracked_vehicle_simulator.getRobotState()
            )

            # ----------------------------------------------
            # 5.6) Update desired 3D pose for logging_utils/debug
            # ----------------------------------------------
            if hasattr(simulator.tracked_vehicle_simulator, "consider_robot_height"):
                pose_des = pose_des.copy()
                pose_des += (
                    simulator.tracked_vehicle_simulator.consider_robot_height
                    * simulator.tracked_vehicle_simulator.w_com_height_vector
                )

            simulator.basePoseW_des = np.concatenate(
                (
                    pose_des,
                    np.array(
                        [
                            terrain_roll_des,
                            terrain_pitch_des,
                            simulator.des_yaw,
                        ]
                    ),
                )
            )

            # ----------------------------------------------
            # 5.7) Optional RViz transform update
            # ----------------------------------------------
            if getattr(simulator, "DEBUG", False):
                self._publish_debug_transform(simulator)

            # ----------------------------------------------
            # 5.8) Get terramechanics interaction forces
            # ----------------------------------------------
            Fx_l, Fx_r, Fy_l, Fy_r = (
                simulator.tracked_vehicle_simulator.getTerramechanicsInteractions()
            )

            # ----------------------------------------------
            # 5.9) Estimate slippages and body velocity
            # ----------------------------------------------
            (
                simulator.beta_l,
                simulator.beta_r,
                simulator.alpha,
                _,
                b_vel_xy,
            ) = simulator.estimateSlippages(
                simulator.baseTwistW,
                simulator.basePoseW[simulator.u.sp_crd["AZ"]],
                simulator.qd_des,
            )

            b_v_x = b_vel_xy[0]
            b_v_y = b_vel_xy[1]

            # ----------------------------------------------
            # 5.10) Body angular velocity around z in base frame
            # ----------------------------------------------
            w_R_b = simulator.math_utils.eul2Rot(
                simulator.u.angPart(simulator.basePoseW)
            )

            b_ang_vel = w_R_b.T.dot(
                simulator.u.angPart(simulator.baseTwistW)
            )

            omega_b = b_ang_vel[2]

            # ----------------------------------------------
            # 5.11) Track-center longitudinal velocities
            # ----------------------------------------------
            v_track_l = b_v_x - omega_b * B / 2.0
            v_track_r = b_v_x + omega_b * B / 2.0

            # ----------------------------------------------
            # 5.12) Total terrain-interaction power
            # ----------------------------------------------
            power = self.compute_power(
                Fx_l=Fx_l,
                Fx_r=Fx_r,
                Fy_l=Fy_l,
                Fy_r=Fy_r,
                v_track_l=v_track_l,
                v_track_r=v_track_r,
                b_v_y=b_v_y,
            )

            if self.positive_only:
                power = max(0.0, power)

            energy += power * simulator.dt

            # ----------------------------------------------
            # 5.13) Store one cost per planning interval
            # ----------------------------------------------
            if simulator.traj.get_knot_transition():
                cost_vector.append(energy - energy_last)
                energy_last = energy

            # ----------------------------------------------
            # 5.14) Stop condition
            # ----------------------------------------------
            if traj_finished:
                if getattr(simulator, "DEBUG", False):
                    print(
                        f"Total energy: {energy} = "
                        f"sum(cost_vector): {sum(cost_vector)}, "
                        f"len(cost_vector): {len(cost_vector)}"
                    )
                break

            # ----------------------------------------------
            # 5.15) Optional logging_utils
            # ----------------------------------------------
            if getattr(simulator, "DEBUG", False):
                if hasattr(simulator, "logData"):
                    simulator.logData()

            # ----------------------------------------------
            # 5.16) Advance simulation time
            # ----------------------------------------------
            simulator.time = np.round(
                simulator.time + np.array([simulator.dt]),
                4,
            )

        return np.asarray(cost_vector, dtype=float)

    def compute_power(
        self,
        Fx_l,
        Fx_r,
        Fy_l,
        Fy_r,
        v_track_l,
        v_track_r,
        b_v_y,
    ):
        """
        Compute total terrain-interaction power.

        P = Fx_L * v_track_L
          + Fx_R * v_track_R
          + Fy_L * v_y
          + Fy_R * v_y
        """

        P_long_left = Fx_l * v_track_l
        P_long_right = Fx_r * v_track_r

        P_lat_left = Fy_l * b_v_y
        P_lat_right = Fy_r * b_v_y

        P_total = (
            P_long_left
            + P_long_right
            + P_lat_left
            + P_lat_right
        )

        return float(P_total)

    def compute_scalar_cost(self, trajectory_reference, simulator):
        """
        Convenience method if CHOMP needs one scalar objective.
        """

        cost_vector = np.asarray(
            self.compute_cost_vector(
                trajectory_reference=trajectory_reference,
                simulator=simulator,
            ),
            dtype=float,
        )

        if cost_vector.size == 0:
            return 0.0

        if not np.all(np.isfinite(cost_vector)):
            cost_vector = np.nan_to_num(
                cost_vector,
                nan=0.0,
                posinf=1e12,
                neginf=-1e12,
            )

        return float(np.sum(cost_vector))

    def _unpack_trajectory_reference(self, trajectory_reference):
        """
        Accept either an object-like trajectory reference or a dictionary.

        Required fields:
            x
            y
            yaw
            v
            omega
            plan_dt or dt
        """

        if isinstance(trajectory_reference, dict):
            des_x_vec = trajectory_reference["x"]
            des_y_vec = trajectory_reference["y"]
            des_yaw_vec = trajectory_reference["yaw"]
            v_ol = trajectory_reference["v"]
            omega_ol = trajectory_reference["omega"]

            plan_dt = trajectory_reference.get(
                "plan_dt",
                trajectory_reference.get("dt", None),
            )

        else:
            des_x_vec = getattr(trajectory_reference, "x")
            des_y_vec = getattr(trajectory_reference, "y")
            des_yaw_vec = getattr(trajectory_reference, "yaw")
            v_ol = getattr(trajectory_reference, "v")
            omega_ol = getattr(trajectory_reference, "omega")

            plan_dt = getattr(
                trajectory_reference,
                "plan_dt",
                getattr(trajectory_reference, "dt", None),
            )

        if plan_dt is None:
            raise ValueError(
                "trajectory_reference must contain either 'plan_dt' or 'dt'."
            )

        des_x_vec = np.asarray(des_x_vec, dtype=float)
        des_y_vec = np.asarray(des_y_vec, dtype=float)
        des_yaw_vec = np.asarray(des_yaw_vec, dtype=float)
        v_ol = np.asarray(v_ol, dtype=float)
        omega_ol = np.asarray(omega_ol, dtype=float)

        n = len(des_x_vec)

        if not (
            len(des_y_vec) == n
            and len(des_yaw_vec) == n
            and len(v_ol) == n
            and len(omega_ol) == n
        ):
            raise ValueError(
                "Trajectory arrays must have the same length: "
                f"len(x)={len(des_x_vec)}, "
                f"len(y)={len(des_y_vec)}, "
                f"len(yaw)={len(des_yaw_vec)}, "
                f"len(v)={len(v_ol)}, "
                f"len(omega)={len(omega_ol)}"
            )

        return (
            des_x_vec,
            des_y_vec,
            des_yaw_vec,
            v_ol,
            omega_ol,
            float(plan_dt),
        )

    def _publish_debug_transform(self, simulator):
        """
        Publish TF in RViz if the simulator has the required ROS objects.
        """

        if not hasattr(simulator, "broadcaster"):
            return

        try:
            import pinocchio as pin

            simulator.now = ros.Time.from_sec(float(simulator.time))

            simulator.euler = simulator.u.angPart(simulator.basePoseW)
            simulator.quaternion = pin.Quaternion(
                pin.rpy.rpyToMatrix(simulator.euler)
            )

            simulator.broadcaster.sendTransform(
                simulator.u.linPart(simulator.basePoseW),
                simulator.quaternion,
                simulator.now,
                "/base_link",
                "/world",
            )

        except Exception as exc:
            print(f"Warning: could not publish debug transform: {exc}")
