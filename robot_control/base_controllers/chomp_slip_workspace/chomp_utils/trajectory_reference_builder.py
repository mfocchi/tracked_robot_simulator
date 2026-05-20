import numpy as np

def build_reference_from_xy(path_m, dt):
    dx_yaw = np.gradient(path_m[:, 0])
    dy_yaw = np.gradient(path_m[:, 1])

    yaw = np.arctan2(dy_yaw, dx_yaw)

    dx = np.diff(path_m[:, 0])
    dy = np.diff(path_m[:, 1])
    dtheta = np.diff(np.unwrap(yaw))

    dx = np.append(dx, dx[-1])
    dy = np.append(dy, dy[-1])
    dtheta = np.append(dtheta, dtheta[-1])

    v = np.hypot(dx, dy) / dt
    omega = dtheta / dt

    return path_m[:, 0], path_m[:, 1], yaw, v, omega


