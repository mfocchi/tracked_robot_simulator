# base_controllers/chomp_slip_workspace/chomp_cost_module/slip_energy_cost_module.py

import numpy as np
import rospy as ros

from base_controllers.tracked_robot.environment.trajectory import Trajectory
from base_controllers.tracked_robot.utils import maxxi_constants as constants

from base_controllers.chomp_slip_workspace.chomp_cost_module.base_cost_module import (
    BaseCostModule,
)


class SlipEnergyCostModule(BaseCostModule):
    """
    Slippage energy cost module.

    Computes the cost:

        P_slip = Fx_L * beta_L
               + Fx_R * beta_R
               + Fy_L * v_y
               + Fy_R * v_y

        J_k = integral of P_slip over one planning interval

    The returned cost is a vector of interval costs, exactly like the old
    EvaluateEnergyConsumption.computeCost(...).
    """

    name = "slip_energy"

    def compute_cost(self, trajectory_reference, simulator):
        """
        Compute slippage energy cost along a trajectory.

        Parameters
        ----------
        trajectory_reference:
            Object containing the reference trajectory.

            Expected attributes:
                x
                y
                yaw
                v
                omega
                plan_dt or dt

        simulator:
            Simulator object compatible with EvaluateEnergyConsumption.

            It must provide:
                initVars()
                initializeSimulation()
                mapFromWorldFrameToBaseFrame(...)
                mapToWheels(...)
                estimateSlippages(...)
                terrainManager
                tracked_vehicle_simulator

        Returns
        -------
        np.ndarray
            Cost vector, one cost per planning interval.
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

        # Make simulator initial/final yaw consistent with CHOMP result.
        # I also set x,y for robustness.
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
        # 2) Optional RViz debug visualization
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
        # 5) Integrate cost over fine simulation time
        # --------------------------------------------------
        cost_vector = []

        energy = 0.0
        energy_last = 0.0

        while not ros.is_shutdown():

            # ----------------------------------------------
            # 5.1) Evaluate desired trajectory at current time
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
            # 5.2) Desired planar velocity -> base-frame velocity
            # ----------------------------------------------
            simulator.b_v_d, simulator.b_omega_d = (
                simulator.mapFromWorldFrameToBaseFrame(
                    simulator.v_d,
                    simulator.omega_d,
                    simulator.basePoseW[3:],
                )
            )

            # ----------------------------------------------
            # 5.3) Desired base velocity -> desired sprocket speeds
            # ----------------------------------------------
            simulator.qd_des = simulator.mapToWheels(
                simulator.b_v_d,
                simulator.b_omega_d,
            )

            # Wheel position integration, mainly for logging_utils/debug.
            simulator.q_des = simulator.q_des + simulator.qd_des * simulator.dt

            # ----------------------------------------------
            # 5.4) Project current robot position onto terrain
            # ----------------------------------------------
            pg, terrain_roll, terrain_pitch = simulator.terrainManager.project_on_mesh(
                point=simulator.basePoseW[:2],
                direction=np.array([0.0, 0.0, 1.0]),
                base_yaw=simulator.basePoseW[5],
            )

            terrain_yaw = simulator.basePoseW[5]

            # Desired pose projected on terrain, useful for logging_utils/debug.
            pose_des, terrain_roll_des, terrain_pitch_des = (
                simulator.terrainManager.project_on_mesh(
                    point=np.array([simulator.des_x, simulator.des_y]),
                    direction=np.array([0.0, 0.0, 1.0]),
                    base_yaw=simulator.des_yaw,
                )
            )

            # ----------------------------------------------
            # 5.5) Simulate one dynamics step
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
            # 5.8) Get terramechanics forces
            # ----------------------------------------------
            Fx_l, Fx_r, Fy_l, Fy_r = (
                simulator.tracked_vehicle_simulator.getTerramechanicsInteractions()
            )

            # ----------------------------------------------
            # 5.9) Estimate slippages
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

            b_v_y = b_vel_xy[1]

            # ----------------------------------------------
            # 5.10) Slip-energy power formula
            # ----------------------------------------------
            power = self.compute_power(
                Fx_l=Fx_l,
                Fx_r=Fx_r,
                Fy_l=Fy_l,
                Fy_r=Fy_r,
                beta_l=simulator.beta_l,
                beta_r=simulator.beta_r,
                b_v_y=b_v_y,
            )

            energy += power * simulator.dt

            # ----------------------------------------------
            # 5.11) Store one cost per CHOMP/planning interval
            # ----------------------------------------------
            if simulator.traj.get_knot_transition():
                cost_vector.append(energy - energy_last)
                energy_last = energy

            # ----------------------------------------------
            # 5.12) Stop condition
            # ----------------------------------------------
            if traj_finished:
                if getattr(simulator, "DEBUG", False):
                    print(
                        f"Slip energy: {energy} = "
                        f"sum(cost_vector): {sum(cost_vector)}, "
                        f"len(cost_vector): {len(cost_vector)}"
                    )
                break

            # ----------------------------------------------
            # 5.13) Optional logging_utils
            # ----------------------------------------------
            if getattr(simulator, "DEBUG", False):
                if hasattr(simulator, "logData"):
                    simulator.logData()

            # ----------------------------------------------
            # 5.14) Advance simulation time
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
        beta_l,
        beta_r,
        b_v_y,
    ):
        """
        Compute instantaneous slip-energy power.

        P = Fx_L beta_L + Fx_R beta_R + Fy_L v_y + Fy_R v_y
        """

        P_long_left = Fx_l * beta_l
        P_long_right = Fx_r * beta_r

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
        Convenience method for CHOMP if it needs one scalar objective.
        """

        cost_vector = self.compute_cost(
            trajectory_reference=trajectory_reference,
            simulator=simulator,
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
            x, y, yaw, v, omega, plan_dt/dt
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