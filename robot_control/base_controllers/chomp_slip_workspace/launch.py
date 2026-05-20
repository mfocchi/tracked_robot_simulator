import numpy as np

from base_controllers.chomp_slip_workspace.chomp_core.chomp_config import ChompConfig
from base_controllers.chomp_slip_workspace.chomp_core.chomp_optimizer import ChompOptimizer
from base_controllers.chomp_slip_workspace.chomp_core.chomp_result import ChompResult

from base_controllers.chomp_slip_workspace.chomp_cost_module import make_cost_module
from base_controllers.chomp_slip_workspace.chomp_gradient_module import make_gradient_module

from base_controllers.chomp_slip_workspace.chomp_utils.world_meter_transform import WorldMeterTransform
from base_controllers.chomp_slip_workspace.chomp_utils.trajectory_reference_builder import (
    build_reference_from_xy,
)


def make_straight_line_initial_path(start_xy_m, goal_xy_m, n_knots):
    """
    Build the initial CHOMP path as a straight line in meters.
    """

    start_xy_m = np.asarray(start_xy_m, dtype=float)
    goal_xy_m = np.asarray(goal_xy_m, dtype=float)

    xi0_m = np.zeros((n_knots, 2), dtype=float)

    xi0_m[:, 0] = np.linspace(start_xy_m[0], goal_xy_m[0], n_knots)
    xi0_m[:, 1] = np.linspace(start_xy_m[1], goal_xy_m[1], n_knots)

    return xi0_m

def setup_cost_module_context(
    cost_module,
    transform,
    X=None,
    Y=None,
    Z=None,
    x_edges=None,
    y_edges=None,
    xRange=None,
    yRange=None,
):
    """
    Give optional external context to the cost module.

    Some costs need terrain data.
    Some costs need world/meter conversion.
    Some costs need neither.

    This keeps chomp_launch generic.
    """

    if hasattr(cost_module, "set_world_meter_transform"):
        try:
            cost_module.set_world_meter_transform(transform)
        except TypeError:
            cost_module.set_world_meter_transform(
                x_min_m=x_edges[0],
                x_max_m=x_edges[-1],
                y_min_m=y_edges[0],
                y_max_m=y_edges[-1],
                xRange=xRange,
                yRange=yRange,
            )

    if hasattr(cost_module, "set_terrain_height_grid"):
        if X is None or Y is None or Z is None:
            raise ValueError(
                f"Cost module '{cost_module.name}' requires terrain grid X, Y, Z."
            )

        cost_module.set_terrain_height_grid(X, Y, Z)


def chomp_launch(
    start_xy_m,
    goal_xy_m,
    X=None,
    Y=None,
    Z=None,
    x_edges=None,
    y_edges=None,
    cost_name="terrain_geometry",
    gradient_name="finite_difference",
    config=None,
):
    """
    Compact modular CHOMP launcher.

    This function only assembles the modules and runs the optimizer.

    It does:
        1. create config
        2. create world-meter transform
        3. create cost module
        4. create gradient module
        5. build initial trajectory
        6. run optimizer
        7. convert result back to meters
        8. build x, y, yaw, v, omega reference
    """

    if config is None:
        config = ChompConfig()

    start_xy_m = np.asarray(start_xy_m, dtype=float)
    goal_xy_m = np.asarray(goal_xy_m, dtype=float)

    # --------------------------------------------------
    # 1) CHOMP internal coordinate range
    # --------------------------------------------------
    xRange = np.array([0.0, 500.0])
    yRange = np.array([0.0, 500.0])

    # --------------------------------------------------
    # 2) Build world-meter transform
    # --------------------------------------------------
    if x_edges is None or y_edges is None:
        raise ValueError(
            "x_edges and y_edges are required to define the world-meter transform."
        )

    transform = WorldMeterTransform(
        x_min_m=x_edges[0],
        x_max_m=x_edges[-1],
        y_min_m=y_edges[0],
        y_max_m=y_edges[-1],
        xRange=xRange,
        yRange=yRange,
    )

    # --------------------------------------------------
    # 3) Create cost and gradient modules
    # --------------------------------------------------
    cost_module = make_cost_module(cost_name)
    gradient_module = make_gradient_module(gradient_name)

    setup_cost_module_context(
        cost_module=cost_module,
        transform=transform,
        X=X,
        Y=Y,
        Z=Z,
        x_edges=x_edges,
        y_edges=y_edges,
        xRange=xRange,
        yRange=yRange,
    )

    # --------------------------------------------------
    # 4) Initial trajectory in meters
    # --------------------------------------------------
    xi0_m = make_straight_line_initial_path(
        start_xy_m=start_xy_m,
        goal_xy_m=goal_xy_m,
        n_knots=config.n_knots,
    )

    # --------------------------------------------------
    # 5) Convert initial trajectory to CHOMP/world units
    # --------------------------------------------------
    xi0_world = transform.meters_to_world_xy(xi0_m)

    # --------------------------------------------------
    # 6) Run modular CHOMP optimizer
    # --------------------------------------------------
    optimizer = ChompOptimizer(
        config=config,
        cost_module=cost_module,
        gradient_module=gradient_module,
    )

    xi_world, cost_history, history_world = optimizer.optimize(xi0_world)

    # --------------------------------------------------
    # 7) Convert optimized trajectory back to meters
    # --------------------------------------------------
    xi_m = transform.world_to_meters_xy(xi_world)

    history_m = [
        transform.world_to_meters_xy(path_w)
        for path_w in history_world
    ]

    # --------------------------------------------------
    # 8) Build full reference trajectory
    # --------------------------------------------------
    x, y, yaw, v, omega = build_reference_from_xy(
        path_m=xi_m,
        dt=config.dt,
    )

    return ChompResult(
        xi_world=xi_world,
        xi_meters=xi_m,
        x=x,
        y=y,
        yaw=yaw,
        v=v,
        omega=omega,
        dt=config.dt,
        cost_history=cost_history,
        trajectory_history_m=history_m,
        cost_name=cost_module.name,
        gradient_name=gradient_module.name,
    )