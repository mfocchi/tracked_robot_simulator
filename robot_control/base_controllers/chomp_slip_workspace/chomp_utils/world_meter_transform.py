import numpy as np

class WorldMeterTransform:
    def __init__(
        self,
        x_min_m,
        x_max_m,
        y_min_m,
        y_max_m,
        xRange,
        yRange,
    ):
        self.xRange = np.asarray(xRange, dtype=float)
        self.yRange = np.asarray(yRange, dtype=float)

        self.x_origin_m = float(x_min_m)
        self.y_origin_m = float(y_min_m)

        self.sx = (x_max_m - x_min_m) / (self.xRange[1] - self.xRange[0])
        self.sy = (y_max_m - y_min_m) / (self.yRange[1] - self.yRange[0])

    def meters_to_world_xy(self, path_m):
        path_m = np.asarray(path_m, dtype=float)

        path_w = path_m.copy()
        path_w[..., 0] = (path_m[..., 0] - self.x_origin_m) / self.sx
        path_w[..., 1] = (path_m[..., 1] - self.y_origin_m) / self.sy

        return path_w

    def world_to_meters_xy(self, path_w):
        path_w = np.asarray(path_w, dtype=float)

        path_m = path_w.copy()
        path_m[..., 0] = self.x_origin_m + path_w[..., 0] * self.sx
        path_m[..., 1] = self.y_origin_m + path_w[..., 1] * self.sy

        return path_m