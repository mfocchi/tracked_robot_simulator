class CostSimulatorAdapter:
    """
    Compatibility wrapper for simulation_workspace.GenericSimulator.

    Legacy cost modules expect a simulator object exposing names like:
        initializeSimulation()
        mapFromWorldFrameToBaseFrame(...)
        mapToWheels(...)
        estimateSlippages(...)
        terrainManager
        tracked_vehicle_simulator

    The newer modular simulator already has the same functionality, but under
    slightly different helpers and nested objects. This adapter bridges those
    differences without changing the optimizer or cost code paths.
    """

    def __init__(self, simulator):
        object.__setattr__(self, "_sim", simulator)

    def __getattr__(self, name):
        if name == "terrainManager":
            return self._terrain_manager()

        if name == "tracked_vehicle_simulator":
            return self._tracked_vehicle_simulator()

        if name == "ros_pub":
            rviz_pub = getattr(self._sim, "rviz_pub", None)
            if rviz_pub is not None:
                return rviz_pub

        return getattr(self._sim, name)

    def __setattr__(self, name, value):
        if name == "_sim":
            object.__setattr__(self, name, value)
            return

        setattr(self._sim, name, value)

    def initVars(self):
        self._ensure_simulator_started()
        self._sim.initVars()

        if hasattr(self._sim, "_init_command_buffers"):
            self._sim._init_command_buffers()

    def initializeSimulation(self):
        self._ensure_simulator_started()
        self._ensure_runtime_helpers()
        self._sim.startupProcedure()

    def mapFromWorldFrameToBaseFrame(self, v_des, omega_des, euler):
        self._ensure_runtime_helpers()

        if getattr(self._sim, "SIMULATOR", None) == "distributed3d":
            return self._sim.wheel_mapper.project_velocity_on_terrain(
                v_des,
                omega_des,
                euler,
            )

        return float(v_des), float(omega_des)

    def mapToWheels(self, b_v_d, b_omega_d):
        self._ensure_runtime_helpers()
        return self._sim.wheel_mapper.map_to_wheels(b_v_d, b_omega_d)

    def estimateSlippages(self, base_twist_w, theta, qd):
        self._ensure_runtime_helpers()
        return self._sim.estimator.estimate(base_twist_w, theta, qd)

    def plotChompTraj3DSpheres(self, *args, **kwargs):
        del args, kwargs

    def _ensure_simulator_started(self):
        if hasattr(self._sim, "startSimulator") and not getattr(
            self._sim,
            "_simulator_started",
            False,
        ):
            self._sim.startSimulator()

        if hasattr(self._sim, "loadModelAndPublishers") and not getattr(
            self._sim,
            "_publishers_loaded",
            False,
        ):
            self._sim.loadModelAndPublishers()

    def _ensure_runtime_helpers(self):
        if not getattr(self._sim, "_vars_initialized", False):
            self._sim.initVars()

        if getattr(self._sim, "wheel_mapper", None) is None:
            raise RuntimeError("Simulator wheel mapper is not initialized.")

        if getattr(self._sim, "estimator", None) is None:
            raise RuntimeError("Simulator slippage estimator is not initialized.")

        if getattr(self._sim, "env_simulator", None) is None:
            raise RuntimeError("Simulator backend is not initialized.")

    def _terrain_manager(self):
        env_simulator = getattr(self._sim, "env_simulator", None)
        terrain_manager = getattr(env_simulator, "terrain_manager", None)

        if terrain_manager is None:
            raise RuntimeError("Terrain manager is not available on the simulator.")

        return terrain_manager

    def _tracked_vehicle_simulator(self):
        env_simulator = getattr(self._sim, "env_simulator", None)
        tracked_vehicle_simulator = getattr(
            env_simulator,
            "tracked_vehicle_simulator",
            None,
        )

        if tracked_vehicle_simulator is None:
            raise RuntimeError(
                "Tracked-vehicle simulator is not available on the simulator."
            )

        return tracked_vehicle_simulator


def wrap_cost_simulator(simulator):
    if simulator is None:
        return None

    if isinstance(simulator, CostSimulatorAdapter):
        return simulator

    has_legacy_interface = all(
        hasattr(simulator, attr)
        for attr in (
            "initializeSimulation",
            "mapFromWorldFrameToBaseFrame",
            "mapToWheels",
            "estimateSlippages",
            "terrainManager",
            "tracked_vehicle_simulator",
        )
    )

    if has_legacy_interface:
        return simulator

    return CostSimulatorAdapter(simulator)
