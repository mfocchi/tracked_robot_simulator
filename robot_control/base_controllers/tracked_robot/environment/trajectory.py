import numpy as np

from base_controllers.tracked_robot.models.unicycle import Unicycle

from enum import Enum
from base_controllers.utils.math_tools import unwrap_angle
from matplotlib import pyplot as plt

class ModelsList(Enum):
    UNICYCLE = 1
    TRACKED = 2
def _show(obj, name):
    print(f"{name}: type={type(obj)}, hasattr(dtype)={hasattr(obj, 'dtype')}, "
          f"dtype={getattr(obj, 'dtype', None)}, shape={getattr(obj, 'shape', None)}, repr={repr(obj)}")



class Trajectory:
    def __init__(self, model=None, start_x=None, start_y=None, start_theta=None, velocity_generator=None, DT = 0.001,  v = None, omega = None,  v_dot = None, omega_dot = None):
        self.start_time = 0.
        self.DT = DT


        if model is not None and np.isscalar(start_x) and  np.isscalar(start_y) and np.isscalar(start_theta):
            self.unwrap = True
            self.x = [start_x]
            self.y = [start_y]
            self.theta = [start_theta]
            self.des_theta_old = 0.
            self.v = []
            self.omega = []
            self.v_dot = []
            self.omega_dot = []
            if model is ModelsList.UNICYCLE:
                self.ideal_unicycle = Unicycle(start_x, start_y, start_theta, DT)
            else:
                assert False, "Trajectory generator: model is not valid"
            if velocity_generator is not None:
                self.init_trajectory(velocity_generator)
            if (v is not None) and (omega is not None):
                if (v_dot is not None) and (omega_dot is not None):
                    self.init_trajectory_with_user_vel(v, omega, v_dot, omega_dot)
                else:
                    self.init_trajectory_with_user_vel(v, omega)
        else: #we are directly providing vectors from externally generated state trajectory start_x, start_y, start_theta are vectors
            self.unwrap = False #no need is responsibility of the user to provide good inputs!
            self.x = start_x
            self.y = start_y
            self.theta = start_theta
            self.v = v
            self.omega = omega
            self.v_dot = np.zeros_like(v)
            self.omega_dot = np.zeros_like(omega)
            self.DT = DT

            assert len(self.x) == len(self.y) == len(self.theta) == len(self.v) == len(self.omega), \
                "Trajectory: All user defined trajectory vectors must have the same length."

    def set_initial_time(self, start_time):
        self.start_time = start_time

    def getSingleUpdate(self, x, y, theta, v, o):
        self.ideal_unicycle.set_state( x, y, theta)
        self.ideal_unicycle.update(v, o)
        return self.ideal_unicycle.x, self.ideal_unicycle.y, self.ideal_unicycle.theta

    def init_trajectory_with_user_vel(self, v, o, v_dot = None, omega_dot = None):
        """
        :param velocity_generator: function which returns a list of longitudinal velocity and a list of angular velocities
        :return: void
        """
        if v_dot is None:
            v_dot = np.zeros_like(v)
        if omega_dot is None:
            omega_dot = np.zeros_like(o)

        assert len(v) == len(o), "Trajectory generator: Invalid input (lenght)"
        for i in range(len(v) - 1):
            self.ideal_unicycle.update(v[i], o[i])
            self.x.append(self.ideal_unicycle.x)
            self.y.append(self.ideal_unicycle.y)
            self.theta.append(self.ideal_unicycle.theta)
            self.v.append(v[i])
            self.omega.append(o[i])
            self.v_dot.append(v_dot[i])
            self.omega_dot.append(omega_dot[i])

        # append finale di velocità nulle
        self.v.append(0.0)
        self.omega.append(0.0)
        self.v_dot.append(0.0)
        self.omega_dot.append(0.0)

    def init_trajectory(self, velocity_generator):
        """
        :param velocity_generator: function which returns a list of longitudinal velocity and a list of angular velocities
        :return: void
        """
        v, o, v_dot, omega_dot, _ = velocity_generator()
        assert len(v) == len(o), "Trajectory generator: Invalid input (lenght)"
        for i in range(len(v) - 1):
            self.ideal_unicycle.update(v[i], o[i])
            self.x.append(self.ideal_unicycle.x)
            self.y.append(self.ideal_unicycle.y)
            self.theta.append(self.ideal_unicycle.theta)
            self.v.append(v[i])
            self.omega.append(o[i])
            self.v_dot.append(v_dot[i])
            self.omega_dot.append(omega_dot[i])
        
        # append finale di velocità nulle
        self.v.append(0.0)
        self.omega.append(0.0)
        self.v_dot.append(0.0)
        self.omega_dot.append(0.0)


    def lerp(self, a, b, alpha):
        return a + alpha * (b - a)

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def lerp_angle(self, theta0, theta1, alpha):
        # shortest angular difference
        d = self.wrap_to_pi(theta1 - theta0)
        return self.wrap_to_pi(theta0 + alpha * d)



    #implemented eval traj that workd for any dt
    def evalTraj(self, current_time):
        #important: to avoid nested arrays force elapsed_time to be a scalar because current_time can be an ndarray
        elapsed_time = float(current_time - self.start_time)

        N = len(self.v)
        if N < 2:
            raise ValueError("Trajectory must have at least 2 points.")

        t_end = (N - 1) * self.DT

        # trajectory duration: last sample at (N)*DT
        if elapsed_time >= t_end:
            print("Trajectory finished")
            return 0, 0, 0, 0, 0, 0, 0, True

        # Continuous index in sample units
        s = elapsed_time / self.DT
        i = int(np.floor(s))
        alpha = s - i  # in [0, 1)

        # numerical safety
        if i < 0:
            i, alpha = 0, 0.0
        elif i >= N - 1:
            i, alpha = N - 2, 1.0

        # Interpolate
        des_x = self.lerp(self.x[i], self.x[i + 1], alpha)
        des_y = self.lerp(self.y[i], self.y[i + 1], alpha)

        # Theta
        des_theta = self.lerp_angle(self.theta[i], self.theta[i + 1], alpha)
        if self.unwrap:
            # keep your original unwrapping behavior if you rely on continuous theta
            des_theta, self.des_theta_old = unwrap_angle(des_theta, self.des_theta_old)

        v_d = self.lerp(self.v[i], self.v[i + 1], alpha)
        omega_d = self.lerp(self.omega[i], self.omega[i + 1], alpha)
        v_dot_d = self.lerp(self.v_dot[i], self.v_dot[i + 1], alpha)
        omega_dot_d = self.lerp(self.omega_dot[i], self.omega_dot[i + 1], alpha)

        return des_x, des_y, des_theta, v_d, omega_d, v_dot_d, omega_dot_d, False


if __name__ == "__main__":
    theta0 = np.pi / 4.0
    q_start = np.array([0.0, 0.0, theta0])
    q_goal = np.array([300.0, 200.0, theta0])
    tf = 40.0
    dT_plan=1
    dT_control = 0.01
    Nsamples_plan = int(tf / dT_plan)

    #create traj
    des_x_vec = np.linspace(q_start[0], q_goal[0], Nsamples_plan)
    des_y_vec = np.linspace(q_start[1], q_goal[1], Nsamples_plan)
    des_theta_vec = np.linspace(q_start[2], q_goal[2], Nsamples_plan)
    des_v_vec = np.linspace(0., 0., Nsamples_plan)
    des_omega_vec = np.linspace(0., 0., Nsamples_plan)
    #init vars
    time = 0.
    log_counter = 0
    des_x_log = np.full(10000, np.nan)
    des_y_log = np.full(10000, np.nan)
    des_theta_log = np.full(10000, np.nan)
    time_log = np.full(10000, np.nan)

    traj = Trajectory(None, des_x_vec, des_y_vec, des_theta_vec, None, DT=dT_plan, v=des_v_vec, omega=des_omega_vec)
    traj.set_initial_time(start_time=time)

    while True:
        des_x, des_y, des_theta, v_d, omega_d, v_dot_d, omega_dot_d, traj_finished = traj.evalTraj(time)
        des_x_log[log_counter] = des_x
        des_y_log[log_counter] = des_y
        des_theta_log[log_counter] = des_theta
        time_log[log_counter]=time
        log_counter+=1
        if np.mod(time, 1) == 0:
            print(f"TIME: {time}")
        time = np.round(time + np.array(dT_control), 4)  # to avoid issues of dt 0.0009999
        if traj_finished:
            break
    #plot
    # xy plot
    plt.figure()
    plt.plot(des_x_log, des_y_log, "-bo", label="interpolated")
    plt.plot(des_x_vec, des_y_vec, "-ro", label="planned",markersize=10, alpha=0.5)
    plt.legend()
    plt.title(f"XY plot:")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(time_log, des_theta_log, "-bo", label="interpolated")
    plt.plot(range(Nsamples_plan), des_theta_vec, "-ro", label="planned", markersize=10, alpha=0.5)
    plt.legend()
    plt.title(f"theta plot:")
    plt.xlabel("time[s]")
    plt.ylabel("theta[m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()