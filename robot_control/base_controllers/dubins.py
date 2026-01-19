import math
import matplotlib
matplotlib.use('TkAgg')
import sys, os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 10000, suppress = True)
# to allow relative import when you call the script directly
# add the parent folder of this file to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unicycle import Unicycle


TWOPI = 2.0 * math.pi


# Data structures
#   Create a structure representing an arc of a Dubins curve (straight or circular)
class DubinsArc:
    def __init__(self, x0, y0, th0, k, L):
        self.L = L
        self.x0 = x0
        self.y0 = y0
        self.th0 = th0
        self.k = k  #curvature
        self.xf, self.yf, self.thf = circline(L, x0, y0, th0, k)

#Create a structure representing a Dubins curve (composed by three arcs)
class DubinsCurve:
    def __init__(self, x0, y0, th0, s1, s2, s3, k0, k1, k2):
        self.a1 = DubinsArc(x0, y0, th0, k0, s1)
        self.a2 = DubinsArc(self.a1.xf, self.a1.yf, self.a1.thf, k1, s2)
        self.a3 = DubinsArc(self.a2.xf, self.a2.yf, self.a2.thf, k2, s3)
        self.L  = self.a1.L + self.a2.L + self.a3.L


def dubins_shortest_path(x0, y0, th0, xf, yf, thf, Kmax):
    # Compute params of standard scaled problem
    sc_th0, sc_thf, sc_Kmax, lambda_ = scaleToStandard(x0, y0, th0, xf, yf, thf, Kmax)

    # Define primitives and curvature signs
    primitives = [LSL, RSR, LSR, RSL, RLR, LRL]
    ksigns = [
        [ 1,  0,  1],  # LSL
        [-1,  0, -1],  # RSR
        [ 1,  0, -1],  # LSR
        [-1,  0,  1],  # RSL
        [-1,  1, -1],  # RLR
        [ 1, -1,  1],  # LRL
    ]

    # Try all primitives, pick optimal
    pidx = -1
    L = math.inf
    sc_s1 = sc_s2 = sc_s3 = 0.0
    for i, primitive in enumerate(primitives):
        ok, sc_s1_c, sc_s2_c, sc_s3_c = primitive(sc_th0, sc_thf, sc_Kmax)
        Lcur = sc_s1_c + sc_s2_c + sc_s3_c
        if ok and Lcur < L:
            L = Lcur
            sc_s1, sc_s2, sc_s3 = sc_s1_c, sc_s2_c, sc_s3_c
            pidx = i

    curve = None
    if pidx >= 0:
        # Scale back to original problem
        s1, s2, s3 = scaleFromStandard(lambda_, sc_s1, sc_s2, sc_s3)

        k1 = ksigns[pidx][0] * Kmax  # this is the curvature = omega/v
        k2 = ksigns[pidx][1] * Kmax  # this is the curvature = omega/v
        k3 = ksigns[pidx][2] * Kmax  # this is the curvature = omega/v

        # Build Dubins curve
        curve = DubinsCurve(
            x0, y0, th0,
            s1, s2, s3,
           k1, k2 , k3
        )

        # Verify correctness
        assert check(sc_s1,
                     ksigns[pidx][0] * sc_Kmax,
                     sc_s2,
                     ksigns[pidx][1] * sc_Kmax,
                     sc_s3,
                     ksigns[pidx][2] * sc_Kmax,
                     sc_th0, sc_thf)

    k = [ks * sc_Kmax for ks in ksigns[pidx]]
    lengths = [curve.a1.L, curve.a2.L, curve.a3.L] if curve else [0, 0, 0]

    return curve, k, lengths


def get_discretized_path_from_dubins(q0, v_max, curve, lengths, dt=None):
    omegas = [0,0,0]
    #get angular velocity from dubins curvature
    omegas[0]=  curve.a1.k *  v_max
    omegas[1] = curve.a2.k * v_max
    omegas[2] = curve.a3.k * v_max

    # initialize model
    unicycle_model = Unicycle(q0[0], q0[1], q0[2], dt)

    # compute total time
    tf = sum(lengths) / v_max

    # switching times (when each arc ends)
    switch_times = np.cumsum(lengths) / v_max

    # initialize outputs
    x_ref = []
    y_ref = []
    theta_ref = []
    v_ref = []
    omega_ref = []
    time = []

    t_ = 0.0
    while t_ <= tf + 1e-12:   # tolerance to include final step
        if t_ < switch_times[0]:
            omega = omegas[0]
        elif t_ < switch_times[1]:
            omega = omegas[1]
        else:
            omega = omegas[2]

        omega_ref.append(omega)
        v_ref.append(v_max)

        unicycle_model.update(v_max, omega)
        x_ref.append(unicycle_model.x)
        y_ref.append(unicycle_model.y)
        theta_ref.append(unicycle_model.theta)
        time.append(t_)

        t_ += dt

    return np.array(x_ref), np.array(y_ref), np.array(theta_ref), np.array(v_ref), np.array(omega_ref),  np.array(time)


# LSL
def LSL(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_thf) - math.cos(sc_th0)
    S = 2 * sc_Kmax + math.sin(sc_th0) - math.sin(sc_thf)
    temp1 = math.atan2(C, S)
    sc_s1 = invK * mod2pi(temp1 - sc_th0)
    temp2 = (2 + 4 * sc_Kmax**2 - 2 * math.cos(sc_th0 - sc_thf) +
             4 * sc_Kmax * (math.sin(sc_th0) - math.sin(sc_thf)))
    if temp2 < 0:
        return False, 0, 0, 0
    sc_s2 = invK * math.sqrt(temp2)
    sc_s3 = invK * mod2pi(sc_thf - temp1)
    return True, sc_s1, sc_s2, sc_s3

# RSR
def RSR(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_th0) - math.cos(sc_thf)
    S = 2 * sc_Kmax - math.sin(sc_th0) + math.sin(sc_thf)
    temp1 = math.atan2(C, S)
    sc_s1 = invK * mod2pi(sc_th0 - temp1)
    temp2 = (2 + 4 * sc_Kmax**2 - 2 * math.cos(sc_th0 - sc_thf) -
             4 * sc_Kmax * (math.sin(sc_th0) - math.sin(sc_thf)))
    if temp2 < 0:
        return False, 0, 0, 0
    sc_s2 = invK * math.sqrt(temp2)
    sc_s3 = invK * mod2pi(temp1 - sc_thf)
    return True, sc_s1, sc_s2, sc_s3

# LSR
def LSR(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_th0) + math.cos(sc_thf)
    S = 2 * sc_Kmax + math.sin(sc_th0) + math.sin(sc_thf)
    temp1 = math.atan2(-C, S)
    temp3 = (4 * sc_Kmax**2 - 2 + 2 * math.cos(sc_th0 - sc_thf) +
             4 * sc_Kmax * (math.sin(sc_th0) + math.sin(sc_thf)))
    if temp3 < 0:
        return False, 0, 0, 0
    sc_s2 = invK * math.sqrt(temp3)
    temp2 = -math.atan2(-2, sc_s2 * sc_Kmax)
    sc_s1 = invK * mod2pi(temp1 + temp2 - sc_th0)
    sc_s3 = invK * mod2pi(temp1 + temp2 - sc_thf)
    return True, sc_s1, sc_s2, sc_s3

# RSL
def RSL(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_th0) + math.cos(sc_thf)
    S = 2 * sc_Kmax - math.sin(sc_th0) - math.sin(sc_thf)
    temp1 = math.atan2(C, S)
    temp3 = (4 * sc_Kmax**2 - 2 + 2 * math.cos(sc_th0 - sc_thf) -
             4 * sc_Kmax * (math.sin(sc_th0) + math.sin(sc_thf)))
    if temp3 < 0:
        return False, 0, 0, 0
    sc_s2 = invK * math.sqrt(temp3)
    temp2 = math.atan2(2, sc_s2 * sc_Kmax)
    sc_s1 = invK * mod2pi(sc_th0 - temp1 + temp2)
    sc_s3 = invK * mod2pi(sc_thf - temp1 + temp2)
    return True, sc_s1, sc_s2, sc_s3

# RLR
def RLR(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_th0) - math.cos(sc_thf)
    S = 2 * sc_Kmax - math.sin(sc_th0) + math.sin(sc_thf)
    temp1 = math.atan2(C, S)
    temp2 = 0.125 * (6 - 4 * sc_Kmax**2 + 2 * math.cos(sc_th0 - sc_thf) +
                     4 * sc_Kmax * (math.sin(sc_th0) - math.sin(sc_thf)))
    if abs(temp2) > 1:
        return False, 0, 0, 0
    sc_s2 = invK * mod2pi(2 * math.pi - math.acos(temp2))
    sc_s1 = invK * mod2pi(sc_th0 - temp1 + 0.5 * sc_s2 * sc_Kmax)
    sc_s3 = invK * mod2pi(sc_th0 - sc_thf + sc_Kmax * (sc_s2 - sc_s1))
    return True, sc_s1, sc_s2, sc_s3

# LRL
def LRL(sc_th0, sc_thf, sc_Kmax):
    invK = 1.0 / sc_Kmax
    C = math.cos(sc_thf) - math.cos(sc_th0)
    S = 2 * sc_Kmax + math.sin(sc_th0) - math.sin(sc_thf)
    temp1 = math.atan2(C, S)
    temp2 = 0.125 * (6 - 4 * sc_Kmax**2 + 2 * math.cos(sc_th0 - sc_thf) -
                     4 * sc_Kmax * (math.sin(sc_th0) - math.sin(sc_thf)))
    if abs(temp2) > 1:
        return False, 0, 0, 0
    sc_s2 = invK * mod2pi(2 * math.pi - math.acos(temp2))
    sc_s1 = invK * mod2pi(temp1 - sc_th0 + 0.5 * sc_s2 * sc_Kmax)
    sc_s3 = invK * mod2pi(sc_thf - sc_th0 + sc_Kmax * (sc_s2 - sc_s1))
    return True, sc_s1, sc_s2, sc_s3


#Functions to scale and solve Dubins problems
def scaleToStandard(x0, y0, th0 ,xf, yf, thf , Kmax):
    dx = (xf - x0)
    dy = (yf - y0)
    assert not (dx==0. and dy==0.), "ERROR: you are sending coincident start/goal to Dubin curve"
    phi = math.atan2(dy, dx)
    lambda_ = math.hypot(dx, dy)/2
    sc_Kmax = Kmax * lambda_
    sc_th0 = mod2pi(th0 - phi)
    sc_thf = mod2pi(thf - phi)
    return sc_th0, sc_thf, sc_Kmax, lambda_

# Scale the solution to the standard problem back to the original problem
def scaleFromStandard(lambda_ , sc_s1, sc_s2, sc_s3):
    s1 = sc_s1 * lambda_
    s2 = sc_s2 * lambda_
    s3 = sc_s3 * lambda_
    return s1, s2, s3

# -------------------- Plotting helpers /Aux functions --------------------

#Normalize an angle (in range [0,2*pi))
def mod2pi(theta: float) -> float:
    out = theta
    while (out < 0):
        out = out + TWOPI
    while (out >= TWOPI):
        out = out - TWOPI
    return out

#Normalize an angular difference (range (-pi, pi])
def rangeSymm(ang):
    out = ang
    while out <= -math.pi:
        out = out + 2 * math.pi
    while out > math.pi:
        out = out - 2 * math.pi
    return out

# Implementation of function sinc(t), returning 1 for t==0, and sin(t)/t otherwise
def sinc(t):
    if (abs(t) < 0.002):
    # For small values of t use Taylor series approximation
        s = 1 - pow(t,2)/6 * (1 - pow(t,2)/20)
    else:
        s = math.sin(t)/t
    return s

# Check validity of a solution by evaluating explicitly the 3 equations  defining a Dubins problem (in standard form)

def check(s1, k0, s2, k1, s3, k2, th0, thf):
    x0 = -1
    y0 = 0
    xf = 1
    yf = 0

    eq1 = (x0
           + s1 * sinc(0.5 * k0 * s1) * math.cos(th0 + 0.5 * k0 * s1)
           + s2 * sinc(0.5 * k1 * s2) * math.cos(th0 + k0 * s1 + 0.5 * k1 * s2)
           + s3 * sinc(0.5 * k2 * s3) * math.cos(th0 + k0 * s1 + k1 * s2 + 0.5 * k2 * s3)
           - xf)

    eq2 = (y0
           + s1 * sinc(0.5 * k0 * s1) * math.sin(th0 + 0.5 * k0 * s1)
           + s2 * sinc(0.5 * k1 * s2) * math.sin(th0 + k0 * s1 + 0.5 * k1 * s2)
           + s3 * sinc(0.5 * k2 * s3) * math.sin(th0 + k0 * s1 + k1 * s2 + 0.5 * k2 * s3)
           - yf)

    eq3 = rangeSymm(k0 * s1 + k1 * s2 + k2 * s3 + th0 - thf)

    Lpos = (s1 > 0) or (s2 > 0) or (s3 > 0)

    result = (math.sqrt(eq1 * eq1 + eq2 * eq2 + eq3 * eq3) < 1e-10) and Lpos
    return result



# Evaluate an arc (circular or straight) composing a Dubins curve, at a  given arc-length s
def circline(s, x0, y0, th0, k):
    x = x0 + s * sinc(k * s / 2.0) * math.cos(th0 + k * s / 2);
    y = y0 + s * sinc(k * s / 2.0) * math.sin(th0 + k * s / 2);
    th = mod2pi(th0 + k * s);
    return x, y, th

def plotarc(arc: DubinsArc, color='b', npts=100):
    pts = []
    for j in range(npts + 1):
        s = arc.L / npts * j
        x, y,_ = circline(s, arc.x0, arc.y0, arc.th0, arc.k)
        pts.append([x, y])
    pts = np.array(pts)
    plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)

def plotdubins(curve: DubinsCurve, color1='r', color2='g', color3='b', show=True):
    plotarc(curve.a1, color1)
    plotarc(curve.a2, color2)
    plotarc(curve.a3, color3)
    # plot initial point
    plt.plot(curve.a1.x0, curve.a1.y0, 'ob')
    # plot final point
    plt.plot(curve.a3.xf, curve.a3.yf, 'ob')
    plt.quiver(curve.a1.x0, curve.a1.y0,0.1 * curve.L * math.cos(curve.a1.th0),0.1 * curve.L * math.sin(curve.a1.th0),
               color='b', angles='xy', scale_units='xy', scale=1, width=0.003)
    plt.quiver(curve.a3.xf, curve.a3.yf, 0.1 * curve.L * math.cos(curve.a3.thf), 0.1 * curve.L * math.sin(curve.a3.thf),
               color='b', angles='xy', scale_units='xy', scale=1, width=0.003)
    plt.grid(True)

    plt.axis('equal')
    if show:
        plt.show(block=False)
    else:
        # ✅ draw immediately even if non-blocking
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    q0 = (0.0, 0.0, -0.3)
    qf = (1.5, 1.0, -1.2)
    Kmax = 5.5
    v_max = 1
    dt = 0.001
    curve, curvatures, lengths = dubins_shortest_path(q0[0], q0[1], q0[2], qf[0], qf[1], qf[2], Kmax)
    x_ref, y_ref, theta_ref, v_ref, omega_ref, time = get_discretized_path_from_dubins(q0, v_max, curve, lengths, dt)
    #plot discretized reference
    plt.plot(x_ref, y_ref, 'k--', label='discr.ref', linewidth=4)
    plt.legend()
    plotdubins(curve, color1='r', color2='g', color3='b')
