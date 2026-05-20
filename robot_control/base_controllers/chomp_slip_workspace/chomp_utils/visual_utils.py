import os
from pathlib import Path

import numpy as np

import matplotlib

# Safer for headless runs, Docker, SSH, ROS machines without display.
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None):
        return text


def _ensure_parent_folder(save_path):
    """
    Create parent folder if save_path is provided.
    """

    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

def compute_terrain_height_grid(
    terrain_manager,
    nx=150,
    ny=150,
    samples_per_cell=1,
    z_margin=5.0,
):
    """
    Discretize the terrain mesh into a regular XY grid and compute
    the terrain height z for each cell.

    Parameters
    ----------
    terrain_manager:
        Object containing:
            terrain_manager.mesh
            terrain_manager.scene

    nx, ny:
        Number of cells in x and y direction.

    samples_per_cell:
        1 means use only the center of each cell.
        3 means sample 3x3 points inside each cell and average them.

    z_margin:
        Height above terrain from which rays are cast downward.

    Returns
    -------
    X, Y, Z, x_edges, y_edges
    """

    import open3d as o3d

    # --------------------------------------------------
    # 1) Get terrain bounds
    # --------------------------------------------------
    bbox = terrain_manager.mesh.get_axis_aligned_bounding_box()

    min_bound = np.asarray(bbox.get_min_bound())
    max_bound = np.asarray(bbox.get_max_bound())

    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound

    # --------------------------------------------------
    # 2) Grid edges and centers
    # --------------------------------------------------
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    Xc, Yc = np.meshgrid(x_centers, y_centers)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    z_origin = z_max + z_margin
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    origins_list = []
    cell_ids_list = []

    # --------------------------------------------------
    # 3) One ray per grid cell
    # --------------------------------------------------
    if samples_per_cell <= 1:
        origins = np.column_stack(
            (
                Xc.ravel(),
                Yc.ravel(),
                np.full(nx * ny, z_origin),
            )
        )

        cell_ids = np.arange(nx * ny)

    # --------------------------------------------------
    # 4) Multiple rays per grid cell
    # --------------------------------------------------
    else:
        offsets = (np.arange(samples_per_cell) + 0.5) / samples_per_cell - 0.5

        x_offsets = offsets * dx
        y_offsets = offsets * dy

        for ox in x_offsets:
            for oy in y_offsets:
                Xs = Xc + ox
                Ys = Yc + oy

                origins_sample = np.column_stack(
                    (
                        Xs.ravel(),
                        Ys.ravel(),
                        np.full(nx * ny, z_origin),
                    )
                )

                origins_list.append(origins_sample)
                cell_ids_list.append(np.arange(nx * ny))

        origins = np.vstack(origins_list)
        cell_ids = np.concatenate(cell_ids_list)

    # --------------------------------------------------
    # 5) Cast rays onto terrain
    # --------------------------------------------------
    directions = np.tile(direction, (origins.shape[0], 1))
    rays_np = np.hstack((origins, directions)).astype(np.float32)

    rays = o3d.core.Tensor(
        rays_np,
        dtype=o3d.core.Dtype.Float32,
    )

    ans = terrain_manager.scene.cast_rays(rays)

    t_hit = ans["t_hit"].numpy()

    hit_z = np.full(origins.shape[0], np.nan)

    valid = np.isfinite(t_hit)
    hit_z[valid] = origins[valid, 2] + t_hit[valid] * direction[2]

    # --------------------------------------------------
    # 6) Average samples inside each grid cell
    # --------------------------------------------------
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

def plot_terrain_height_grid(
    X,
    Y,
    Z,
    x_edges,
    y_edges,
    save_path=None,
    show=False,
    title="Terrain height map",
    dpi=200,
):
    """
    Plot the terrain height map.

    Parameters
    ----------
    X, Y, Z:
        Terrain grid arrays.

    x_edges, y_edges:
        Grid edges.

    save_path:
        Output image path. If None, the figure is only shown if show=True.

    show:
        If True, open an interactive matplotlib window.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
    )

    fig.colorbar(im, ax=ax, label="terrain height z [m]")

    ax.contour(
        X,
        Y,
        Z,
        levels=20,
        linewidths=0.5,
    )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        _ensure_parent_folder(save_path)
        fig.savefig(save_path, dpi=dpi)
        print(colored(f"Saved terrain height map image to: {save_path}", "green"))

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)

def plot_terrain_height_grid_with_paths(
    X,
    Y,
    Z,
    x_edges,
    y_edges,
    paths_m=None,
    labels=None,
    save_path=None,
    show=False,
    title="Terrain height map with CHOMP path",
    dpi=200,
):
    """
    Plot terrain height heatmap and overlay one or more paths.

    Parameters
    ----------
    paths_m:
        List of paths. Each path must be shape (N, 2), in meters.

    labels:
        List of labels, one for each path.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
    )

    fig.colorbar(im, ax=ax, label="terrain height z [m]")

    ax.contour(
        X,
        Y,
        Z,
        levels=20,
        linewidths=0.5,
    )

    if paths_m is not None:
        if labels is None:
            labels = [f"path {i}" for i in range(len(paths_m))]

        for path, label in zip(paths_m, labels):
            path = np.asarray(path, dtype=float)

            if path.ndim != 2 or path.shape[1] < 2:
                raise ValueError(
                    f"Each path must have shape (N, 2) or (N, >=2), got {path.shape}"
                )

            ax.plot(
                path[:, 0],
                path[:, 1],
                "-o",
                linewidth=2,
                markersize=3,
                label=label,
            )

            ax.plot(
                path[0, 0],
                path[0, 1],
                "go",
                markersize=8,
                label=None,
            )

            ax.plot(
                path[-1, 0],
                path[-1, 1],
                "rx",
                markersize=8,
                label=None,
            )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)

    if paths_m is not None:
        ax.legend()

    fig.tight_layout()

    if save_path is not None:
        _ensure_parent_folder(save_path)
        fig.savefig(save_path, dpi=dpi)
        print(colored(f"Saved terrain height map with paths to: {save_path}", "green"))

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)

def animate_chomp_history_on_terrain(
    X,
    Y,
    Z,
    x_edges,
    y_edges,
    chomp_history_m,
    save_path=None,
    interval=250,
    show=False,
    title="CHOMP iterations on terrain height map",
):
    """
    Animate CHOMP path evolution on top of the terrain height map.

    Parameters
    ----------
    chomp_history_m:
        List of arrays. Each array must have shape (N, 2), in meters.

    save_path:
        Output GIF path.

    interval:
        Delay between frames in milliseconds.
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
        aspect="equal",
    )

    fig.colorbar(im, ax=ax, label="terrain height z [m]")

    ax.contour(
        X,
        Y,
        Z,
        levels=20,
        linewidths=0.5,
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
        bbox=dict(facecolor="white", alpha=0.7),
    )

    ax.set_title(title)
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
        path = np.asarray(chomp_history_m[frame_idx], dtype=float)

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
        blit=True,
    )

    if save_path is not None:
        _ensure_parent_folder(save_path)

        fps = max(1, int(1000 / interval))

        anim.save(
            save_path,
            writer=PillowWriter(fps=fps),
        )

        print(colored(f"Saved CHOMP iteration animation to: {save_path}", "green"))

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)

def save_chomp_history_frames_on_terrain(
    X,
    Y,
    Z,
    x_edges,
    y_edges,
    chomp_history_m,
    output_folder,
    frame_stride=1,
    dpi=200,
):
    """
    Save one PNG image per CHOMP iteration.
    """

    if chomp_history_m is None or len(chomp_history_m) == 0:
        print(colored("No CHOMP history to save.", "red"))
        return

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(0, len(chomp_history_m), frame_stride):
        path = np.asarray(chomp_history_m[frame_idx], dtype=float)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal",
        )

        fig.colorbar(im, ax=ax, label="terrain height z [m]")

        ax.contour(
            X,
            Y,
            Z,
            levels=20,
            linewidths=0.5,
        )

        ax.plot(
            path[:, 0],
            path[:, 1],
            "-o",
            linewidth=2,
            markersize=3,
            label=f"iteration {frame_idx}",
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

        frame_path = output_folder / f"chomp_iteration_{frame_idx:04d}.png"

        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)

    print(colored(f"Saved CHOMP iteration frames to: {output_folder}", "green"))

def print_terrain_height_candidates(X, Y, Z, n=10):
    """
    Print the n lowest and n highest terrain cells.

    This is not exactly visualization, but it is useful for choosing
    start and goal points on the terrain.
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

def save_terrain_height_grid(X, Y, Z, folder):
    """
    Save terrain grid as CSV files.

    Saves:
        terrain_height_points.csv
        terrain_height_matrix.csv
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    points_path = folder / "terrain_height_points.csv"
    matrix_path = folder / "terrain_height_matrix.csv"

    data = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
        )
    )

    np.savetxt(
        points_path,
        data,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.6f",
    )

    np.savetxt(
        matrix_path,
        Z,
        delimiter=",",
        fmt="%.6f",
    )

    print(colored(f"Saved terrain point cloud grid to: {points_path}", "green"))
    print(colored(f"Saved terrain height matrix to: {matrix_path}", "green"))

def visualize_chomp_result_on_terrain(
    result,
    X,
    Y,
    Z,
    x_edges,
    y_edges,
    output_dir,
    experiment_name="chomp_test",
    save_height_map=True,
    save_path_plot=True,
    save_gif=True,
    save_frames=False,
    frame_stride=1,
    show=False,
):
    """
    Visualize a CHOMP result on a terrain height map.

    This is a convenience wrapper around the lower-level plotting utilities.
    """

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) Terrain height map
    # --------------------------------------------------
    if save_height_map:
        plot_terrain_height_grid(
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            save_path=output_dir / f"{experiment_name}_terrain_height_map.png",
            show=show,
        )

    # --------------------------------------------------
    # 2) Final path over terrain
    # --------------------------------------------------
    if save_path_plot:
        paths_m = []
        labels = []

        if result.trajectory_history_m is not None and len(result.trajectory_history_m) > 0:
            paths_m.append(result.trajectory_history_m[0])
            labels.append("initial path")

        paths_m.append(result.xi_meters[:, 0:2])
        labels.append("optimized path")

        plot_terrain_height_grid_with_paths(
            X,
            Y,
            Z,
            x_edges,
            y_edges,
            paths_m=paths_m,
            labels=labels,
            save_path=output_dir / f"{experiment_name}_path.png",
            show=show,
        )

    # --------------------------------------------------
    # 3) CHOMP history GIF
    # --------------------------------------------------
    if save_gif:
        if result.trajectory_history_m is not None and len(result.trajectory_history_m) > 0:
            animate_chomp_history_on_terrain(
                X,
                Y,
                Z,
                x_edges,
                y_edges,
                result.trajectory_history_m,
                save_path=output_dir / f"{experiment_name}_iterations.gif",
                interval=250,
                show=show,
            )

    # --------------------------------------------------
    # 4) Individual frames
    # --------------------------------------------------
    if save_frames:
        if result.trajectory_history_m is not None and len(result.trajectory_history_m) > 0:
            save_chomp_history_frames_on_terrain(
                X,
                Y,
                Z,
                x_edges,
                y_edges,
                result.trajectory_history_m,
                output_folder=output_dir / f"{experiment_name}_frames",
                frame_stride=frame_stride,
            )


class TerrainVisualUtils:
    """
    Optional namespace-style wrapper.

    You can either use:
        plot_terrain_height_grid(...)

    or:
        TerrainVisualUtils.plot_terrain_height_grid(...)
    """

    plot_terrain_height_grid = staticmethod(plot_terrain_height_grid)
    plot_terrain_height_grid_with_paths = staticmethod(plot_terrain_height_grid_with_paths)
    animate_chomp_history_on_terrain = staticmethod(animate_chomp_history_on_terrain)
    save_chomp_history_frames_on_terrain = staticmethod(save_chomp_history_frames_on_terrain)
    print_terrain_height_candidates = staticmethod(print_terrain_height_candidates)
    save_terrain_height_grid = staticmethod(save_terrain_height_grid)
    visualize_chomp_result_on_terrain = staticmethod(visualize_chomp_result_on_terrain)

