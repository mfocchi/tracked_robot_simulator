#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shutil, sys
import numpy as np
import rospkg
from pathlib import Path

from base_controllers.tracked_robot.simulator.terrain_manager import TerrainManager

from base_controllers.chomp_slip_workspace.launch import chomp_launch
from base_controllers.chomp_slip_workspace.chomp_core.chomp_config import ChompConfig

from base_controllers.chomp_slip_workspace.chomp_utils.visual_utils import (
    compute_terrain_height_grid,
    print_terrain_height_candidates,
    visualize_chomp_result_on_terrain,
)


def _bootstrap_workspace_parent():
    package_dir = Path(__file__).resolve().parent
    parent_dir = package_dir.parent

    for path in (str(parent_dir), str(package_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_generic_simulator_class():
    _bootstrap_workspace_parent()

    failures = []
    module_names = (
        "simulation_workspace.core.simulator",
        "base_controllers.simulation_workspace.core.simulator",
    )

    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["GenericSimulator"])
            return module.GenericSimulator
        except ModuleNotFoundError as exc:
            failures.append(f"{module_name}: {exc}")

    failure_text = "\n".join(f"  - {failure}" for failure in failures)
    raise RuntimeError(
        "Could not import the simulation workspace GenericSimulator.\n"
        f"{failure_text}"
    )


def _build_cost_simulator(terrain_type):
    GenericSimulator = _load_generic_simulator_class()
    simulator = GenericSimulator("tractor")
    simulator.TERRAIN_TYPE = terrain_type

    if hasattr(simulator, "start"):
        simulator.start()

    simulator.startSimulator()
    simulator.loadModelAndPublishers()
    return simulator

def main():

    # ==================================================
    # 0) EXPERIMENT CONFIGURATION
    # ==================================================

    # ------------------------------
    # Terrain / mesh configuration
    # ------------------------------
    terrain_type = "terrain_chen2"

    mesh_package_name = "tractor_description"
    mesh_folder_name = "meshes"

    terrain_grid_params = dict(
        nx=150,
        ny=150,
        samples_per_cell=1,
        z_margin=5.0,
    )

    print_height_candidates_params = dict(
        n=10,
    )

    # ------------------------------
    # Test cases
    # ------------------------------
    test_cases = [
        (
            "case_1_high_to_low",
            np.array([0.0, 0.0]),
            np.array([10.0, -2.0]),
        ),
        (
            "case_2_low_to_high",
            np.array([10.0, -2.0]),
            np.array([0.0, 0.0]),
        ),
        (
            "case_3_hill_avoidance",
            np.array([-15.0, -12.0]),
            np.array([-10.0, 5.0]),
        ),
    ]

    # ------------------------------
    # CHOMP algorithm configuration
    # ------------------------------
    cost_name = "slip_energy"    # "slip_energy", "total_energy", "terrain_geometry"
    gradient_name = "spsa"  # "spsa", "finite_difference", "analytic"

    chomp_config = ChompConfig(
        dof=2,
        n_knots=40,
        dt=1.0,
        max_iter=100,
        tol=1.0,
        eta=0.001,
        lambda_smooth=200.0,
        save_history=True,
    )

    # ------------------------------
    # Output / visualization configuration
    # ------------------------------

    output_root_dir = os.path.join(
        "/root/ros_ws/src/tracked_robot_simulator/robot_control/base_controllers",
        "chomp_slip_workspace",
        "outputs",
    )

    experiment_group_name = cost_name + "_" + gradient_name

    experiment_output_dir = os.path.join(
        output_root_dir,
        experiment_group_name,
    )

    if os.path.exists(experiment_output_dir):
        shutil.rmtree(experiment_output_dir)

    os.makedirs(experiment_output_dir, exist_ok=True)

    visualization_params = dict(
        save_height_map=False,
        save_path_plot=True,
        save_gif=True,
        save_frames=False,
        frame_stride=1,
        show=False,
    )
    # ==================================================
    # 1) LOAD TERRAIN MANAGER
    # ==================================================

    mesh_path = os.path.join(
        rospkg.RosPack().get_path(mesh_package_name),
        mesh_folder_name,
        terrain_type + ".stl",
    )

    terrain_manager = TerrainManager(mesh_path)
    cost_simulator = None

    if cost_name in {"slip_energy", "total_energy"}:
        cost_simulator = _build_cost_simulator(terrain_type=terrain_type)

    # ==================================================
    # 2) COMPUTE TERRAIN HEIGHT GRID
    # ==================================================

    X, Y, Z, x_edges, y_edges = compute_terrain_height_grid(
        terrain_manager=terrain_manager,
        **terrain_grid_params,
    )

    print_terrain_height_candidates(
        X,
        Y,
        Z,
        **print_height_candidates_params,
    )

    # ==================================================
    # 3) RUN CHOMP TEST CASES
    # ==================================================

    for case_name, start_xy_m, goal_xy_m in test_cases:

        print("\n" + "=" * 80)
        print("Running test case:", case_name)
        print("Start:", start_xy_m)
        print("Goal :", goal_xy_m)
        print("=" * 80)

        result = chomp_launch(
            start_xy_m=start_xy_m,
            goal_xy_m=goal_xy_m,
            X=X,
            Y=Y,
            Z=Z,
            x_edges=x_edges,
            y_edges=y_edges,
            cost_name=cost_name,
            gradient_name=gradient_name,
            simulator=cost_simulator,
            config=chomp_config,
        )

        # ==================================================
        # 4) PRINT RESULT SUMMARY
        # ==================================================

        print("Cost type:", result.cost_name)
        print("Gradient type:", result.gradient_name)
        print("Final cost:", result.cost_history[-1])

        # ==================================================
        # 5) VISUALIZE RESULT
        # ==================================================

        output_dir = os.path.join(
            experiment_output_dir,
            case_name,
        )

        visualize_chomp_result_on_terrain(
            result=result,
            X=X,
            Y=Y,
            Z=Z,
            x_edges=x_edges,
            y_edges=y_edges,
            output_dir=output_dir,
            experiment_name= cost_name + "_" + gradient_name + "_" + case_name,
            **visualization_params,
        )

if __name__ == "__main__":
    main()
