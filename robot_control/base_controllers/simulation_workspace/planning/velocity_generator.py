#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Velocity profile generator for smooth acceleration/deceleration.
"""
import numpy as np
from scipy import signal


class VelocityGenerator:
    """
    Generates smooth velocity profiles using sigmoid functions.

    Creates velocity commands that respect acceleration limits
    and provide smooth transitions between velocity states.
    """

    def __init__(self, simulation_time=20.0, DT=0.001):
        """
        Args:
            simulation_time: Total simulation duration [s]
            DT: Time step [s]
        """
        self.simulation_time = simulation_time
        self.DT = DT
        self.n_steps = int(np.floor(simulation_time / DT))

        # Time vector
        self.time = np.linspace(0, simulation_time, self.n_steps)

    def velocity_mir_smooth(self, v_max_=0.4, omega_max_=0.2):
        """
        Generate smooth velocity profiles with sigmoid transitions.

        Creates a velocity profile that:
        1. Accelerates smoothly to max forward velocity
        2. Maintains max velocity
        3. Decelerates smoothly to zero
        4. Accelerates smoothly to max angular velocity (turn)
        5. Maintains turn
        6. Decelerates smoothly to zero

        Args:
            v_max_: Maximum linear velocity [m/s]
            omega_max_: Maximum angular velocity [rad/s]

        Returns:
            tuple: (v, omega, v_dot, omega_dot, time)
        """
        print(f"Generating velocity profile: v_max={v_max_}, omega_max={omega_max_}")

        # Define transition points (as fraction of total time)
        transitions = {
            'accel_start': 0.1,
            'accel_end': 0.2,
            'cruise_end': 0.5,
            'decel_start': 0.5,
            'decel_end': 0.6,
            'turn_accel_start': 0.7,
            'turn_accel_end': 0.8,
            'turn_end': 0.9,
            'turn_decel_end': 1.0
        }

        # Create sigmoid transition functions
        def sigmoid_transition(t, t_start, t_end, max_val):
            """Smooth sigmoid transition between 0 and max_val."""
            mid = (t_start + t_end) / 2
            steepness = 10.0 / (t_end - t_start)  # Adjust steepness based on duration
            return max_val / (1 + np.exp(-steepness * (t - mid)))

        # Generate linear velocity profile
        v = np.zeros(self.n_steps)

        # Acceleration phase
        accel_mask = (self.time >= transitions['accel_start']) & \
                     (self.time <= transitions['accel_end'])
        t_accel = self.time[accel_mask]
        v[accel_mask] = sigmoid_transition(
            t_accel,
            transitions['accel_start'],
            transitions['accel_end'],
            v_max_
        )

        # Cruise phase
        cruise_mask = (self.time > transitions['accel_end']) & \
                      (self.time <= transitions['decel_start'])
        v[cruise_mask] = v_max_

        # Deceleration phase
        decel_mask = (self.time > transitions['decel_start']) & \
                     (self.time <= transitions['decel_end'])
        t_decel = self.time[decel_mask]
        v[decel_mask] = v_max_ - sigmoid_transition(
            t_decel,
            transitions['decel_start'],
            transitions['decel_end'],
            v_max_
        )

        #