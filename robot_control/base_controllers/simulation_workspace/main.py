#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path


def _bootstrap_local_package():
    """
    Make sure the parent directory of this package is importable before we
    resolve the simulator module.

    When this file is executed directly, Python adds the package directory
    itself to ``sys.path`` but not necessarily its parent, which is required to
    import ``simulation_workspace.*`` as a package.
    """
    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent

    for path in (str(parent_dir), str(package_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    return package_dir.name


def _load_ros():
    try:
        import rospy as ros
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing ROS Python dependency 'rospy'. This workspace can be cleaned up statically here, "
            "but it cannot run end-to-end without the ROS environment."
        ) from exc
    return ros


def _load_simulator_class():
    local_package_name = _bootstrap_local_package()
    candidates = tuple(dict.fromkeys((
        f"{local_package_name}.core.simulator",
        "simulation_workspace.core.simulator",
        "simulation_workspace.core.simulator",
        "base_controllers.simulation_workspace.core.simulator",
        "base_controllers.simulation_workspace.core.simulator",
    )))
    failures = []

    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["GenericSimulator"])
            return module.GenericSimulator
        except ModuleNotFoundError as exc:
            failures.append(f"{module_name}: {exc}")

    failure_text = "\n".join(f"  - {failure}" for failure in failures)
    raise RuntimeError(
        "Could not import GenericSimulator from either the local package or the legacy framework path.\n"
        f"{failure_text}"
    )


def main():
    sim = None

    try:
        ros = _load_ros()
        GenericSimulator = _load_simulator_class()
        sim = GenericSimulator("tractor")

        # Choose one test case here:
        # sim.set_chomp_test_case("hill_avoidance")
        sim.set_chomp_test_case("high_to_low")
        # sim.set_chomp_test_case("low_to_high")
        # sim.set_chomp_test_case("through_two_hills")

        sim.start()
        sim.startSimulator()
        sim.loadModelAndPublishers()

        if sim.ControlType == "OPEN_LOOP":
            sim.run_open_loop()
        else:
            sim.run_closed_loop()
        return 0
    except RuntimeError as exc:
        print(exc)
        return 1
    except (ros.ROSInterruptException, ros.service.ServiceException):
        return 0
    finally:
        if sim is not None:
            if sim.SAVE_BAGS and hasattr(sim, "recorder") and hasattr(sim.recorder, "stop_recording_srv"):
                try:
                    sim.recorder.stop_recording_srv()
                except Exception:
                    pass
            try:
                ros.signal_shutdown("Simulation finished")
            except Exception:
                pass
            if hasattr(sim, "deregister_node"):
                try:
                    sim.deregister_node()
                except Exception:
                    pass
            if hasattr(sim, "plotData"):
                print("Plotting results...")
                sim.plotData()


if __name__ == "__main__":
    sys.exit(main())
