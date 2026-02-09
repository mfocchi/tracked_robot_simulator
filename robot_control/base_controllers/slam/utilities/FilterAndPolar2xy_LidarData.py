import numpy as np
import matplotlib.pyplot as plt

def filterAndPolar2XY_LidarData(msg):
    points = []
    angle = msg.angle_min
    for d in msg.ranges:
        if np.isfinite(d): # discard inf and keep only distances
            x = d * np.cos(angle)
            y = d * np.sin(angle)
            points.append([x, y])
        angle += msg.angle_increment

    return np.array(points)

# no tused in the end, too heavy, mismatch frequency#
"""
def plot_lidar_points(pointsXY, ax):
    if pointsXY.shape[0] == 0:
        return

    ax.clear()
    ax.scatter(pointsXY[:, 0], pointsXY[:, 1], s=8)
    ax.set_title("LiDAR points (frame lidar)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)

    plt.pause(0.001)
"""