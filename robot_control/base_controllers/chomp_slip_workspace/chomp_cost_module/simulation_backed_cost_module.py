from abc import abstractmethod

import numpy as np

from base_controllers.chomp_slip_workspace.chomp_cost_module.base_cost_module import (
    BaseCostModule,
)
from base_controllers.chomp_slip_workspace.chomp_utils.simulator_cost_adapter import (
    wrap_cost_simulator,
)
from base_controllers.chomp_slip_workspace.chomp_utils.trajectory_reference_builder import (
    build_reference_from_xy,
)


class SimulationBackedCostModule(BaseCostModule):
    """
    Bridge between the generic CHOMP cost API and simulator-backed costs.

    The optimizer always calls costs as:

        compute_cost(xi_xy=<world path>, dt=<plan dt>)

    while legacy simulation-backed costs expect:

        compute_cost(trajectory_reference=<meters path>, simulator=<sim>)

    This base class rebuilds the trajectory reference and resolves the
    simulator context so the optimizer does not need any special cases.
    """

    requires_simulator = True

    def __init__(self):
        self.transform = None
        self.simulator = None

    def set_world_meter_transform(self, transform):
        if transform is None:
            raise ValueError("transform cannot be None.")

        if not hasattr(transform, "world_to_meters_xy"):
            raise TypeError("transform must provide world_to_meters_xy(...).")

        if not hasattr(transform, "meters_to_world_xy"):
            raise TypeError("transform must provide meters_to_world_xy(...).")

        self.transform = transform

    def set_simulator(self, simulator):
        self.simulator = wrap_cost_simulator(simulator)

    def world_to_meters_xy(self, path_w):
        if self.transform is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a world/meter transform. "
                "Call set_world_meter_transform(...) first."
            )

        return self.transform.world_to_meters_xy(path_w)

    def build_trajectory_reference(self, xi_xy, dt):
        xi_xy = np.asarray(xi_xy, dtype=float)

        if xi_xy.ndim != 2:
            raise ValueError(f"xi_xy must be 2D, got shape {xi_xy.shape}.")

        if xi_xy.shape[1] != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects xi_xy with shape (T, 2), "
                f"got {xi_xy.shape}."
            )

        path_m = self.world_to_meters_xy(xi_xy)
        dt = float(dt)

        x, y, yaw, v, omega = build_reference_from_xy(
            path_m=path_m,
            dt=dt,
        )

        return {
            "x": x,
            "y": y,
            "yaw": yaw,
            "v": v,
            "omega": omega,
            "dt": dt,
            "plan_dt": dt,
        }

    def compute_cost(self, xi_xy=None, dt=None, **kwargs):
        trajectory_reference, simulator = self._resolve_compute_inputs(
            xi_xy=xi_xy,
            dt=dt,
            **kwargs,
        )

        return self.compute_scalar_cost(
            trajectory_reference=trajectory_reference,
            simulator=simulator,
        )

    def _resolve_compute_inputs(self, xi_xy=None, dt=None, **kwargs):
        trajectory_reference = kwargs.pop("trajectory_reference", None)
        simulator = kwargs.pop("simulator", None)

        if trajectory_reference is None and self._looks_like_trajectory_reference(xi_xy):
            trajectory_reference = xi_xy
            simulator = simulator or dt
            xi_xy = None
            dt = None

        if trajectory_reference is None:
            if xi_xy is None or dt is None:
                raise ValueError(
                    f"{self.__class__.__name__}.compute_cost(...) needs either "
                    "(xi_xy, dt) or (trajectory_reference, simulator)."
                )

            trajectory_reference = self.build_trajectory_reference(
                xi_xy=xi_xy,
                dt=dt,
            )

        simulator = wrap_cost_simulator(simulator) if simulator is not None else self.simulator

        if simulator is None:
            raise RuntimeError(
                f"Cost module '{self.name}' needs a simulator context. "
                "Pass simulator=... to chomp_launch(...) or call "
                "cost_module.set_simulator(...)."
            )

        return trajectory_reference, simulator

    def _looks_like_trajectory_reference(self, candidate):
        required_keys = ("x", "y", "yaw", "v", "omega")

        if isinstance(candidate, dict):
            return all(key in candidate for key in required_keys)

        return candidate is not None and all(
            hasattr(candidate, key) for key in required_keys
        )

    def compute_scalar_cost(self, trajectory_reference, simulator):
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

    @abstractmethod
    def compute_cost_vector(self, trajectory_reference, simulator):
        raise NotImplementedError
