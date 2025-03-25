# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

robot_params = {}



robot_params['tractor'] ={'dt': 0.01,
                        'kp': np.array([100.,   100.,    100.,  100.]),
                        'kd':  np.array([10.,    10.,    10.,   10.  ]),
                        'q_0':  np.array([0, 0, 0, 0]),
                        'joint_names': ['front_left_wheel_joint', 'front_right_wheel_joint'], # caster wheels are passive joints
                        'ee_frames': ['front_left_wheel', 'front_right_wheel'],
                        'spawn_x': -0.,
                        'spawn_y': 0.0,
                        'spawn_z': 1.5,
                        'buffer_size': 7000}

verbose = False
plotting = True


