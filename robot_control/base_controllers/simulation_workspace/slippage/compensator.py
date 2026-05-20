#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feed‑forward slip compensation using ML or analytical models.
"""
import os
import numpy as np
import sys
from termcolor import colored
from base_controllers.tracked_robot.utils import maxxi_constants as constants


class SlipCompensator:
    """
    Applies slip compensation to wheel commands.

    Supports:
    - Machine learning based (CatBoost or RBF interpolator)
    - Exponential model (EXP)
    - None (pass‑through)
    """

    def __init__(self, sim):
        self.sim = sim
        self.models = {}
        self._models_loaded = False

    def init_models(self):
        """Load ML models for slip prediction."""
        if self._models_loaded:
            return
        try:
            slippage_type = self.sim.SLIPPAGE_INFERENCE_TYPE
            fc = self.sim.friction_coefficient
            flag3d = self.sim.flag3D

            if slippage_type == 'decision_trees':
                self._load_catboost_models(fc, flag3d)
            elif slippage_type == 'interpolator':
                self._load_interpolator_models(fc)
            self._models_loaded = True
        except Exception as e:
            print(colored(f"Error loading slip models: {e}", "red"))
            print(colored("Disabling slip compensation and continuing without learned models.", "yellow"))
            self.models = {}
            self._models_loaded = False
            self.sim.SIDE_SLIP_COMPENSATION = 'NONE'
            self.sim.LONG_SLIP_COMPENSATION = 'NONE'

    def _load_catboost_models(self, friction_coeff, flag3d):
        import catboost as cb
        base = os.environ['LOCOSIM_DIR'] + '/robot_control/base_controllers/tracked_robot/regressor/'
        self.models['beta_l'] = cb.CatBoostRegressor().load_model(
            f"{base}model_beta_l{flag3d}{friction_coeff}.cb")
        self.models['beta_r'] = cb.CatBoostRegressor().load_model(
            f"{base}model_beta_r{flag3d}{friction_coeff}.cb")
        self.models['alpha'] = cb.CatBoostRegressor().load_model(
            f"{base}model_alpha{flag3d}{friction_coeff}.cb")

    def _load_interpolator_models(self, friction_coeff):
        from scipy.interpolate import RBFInterpolator
        import pandas as pd
        base = os.environ['LOCOSIM_DIR'] + '/robot_control/base_controllers/tracked_robot/regressor/'
        data_path = f'{base}ident_wheels_sim_2d_{friction_coeff}.csv'
        df = pd.read_csv(data_path, skiprows=1,
                         names=['wheel_l', 'wheel_r', 'beta_l', 'beta_r', 'alpha'])
        x = df[['wheel_l', 'wheel_r']].values
        y = df[['beta_l', 'beta_r', 'alpha']].values
        self.models['beta_l'] = RBFInterpolator(x, y[:, 0], smoothing=0.1)
        self.models['beta_r'] = RBFInterpolator(x, y[:, 1], smoothing=0.1)
        self.models['alpha'] = RBFInterpolator(x, y[:, 2], smoothing=0.1)

    def compensate(self, v, omega, qd_des):
        """Apply slip compensation based on configured method.
        Returns: (qd_comp, beta_l, beta_r)"""
        comp_type = self.sim.LONG_SLIP_COMPENSATION
        if comp_type == 'MACHINE_LEARNING':
            return self._compensate_ml(qd_des)
        elif comp_type == 'EXP':
            return self._compensate_exp(v, omega, qd_des)
        else:
            return qd_des, 0.0, 0.0

    def _compensate_ml(self, qd_des):
        if 'beta_l' not in self.models or 'beta_r' not in self.models:
            return qd_des, 0.0, 0.0

        v_enc_l = constants.SPROCKET_RADIUS * qd_des[0]
        v_enc_r = constants.SPROCKET_RADIUS * qd_des[1]

        if self.sim.SLIPPAGE_INFERENCE_TYPE == 'decision_trees':
            if len(self.models['beta_l'].feature_names_) > 2:
                pose = self.sim.basePoseW
                inp = np.array([qd_des[0], qd_des[1], pose[3], pose[4], pose[5]])
                beta_l = self.models['beta_l'].predict(inp)
                beta_r = self.models['beta_r'].predict(inp)
            else:
                beta_l = self.models['beta_l'].predict(qd_des)
                beta_r = self.models['beta_r'].predict(qd_des)
        else:  # interpolator
            beta_l = self.models['beta_l']([qd_des]).squeeze()
            beta_r = self.models['beta_r']([qd_des]).squeeze()

        v_enc_l += beta_l
        v_enc_r += beta_r

        qd_comp = np.zeros(2)
        qd_comp[0] = v_enc_l / constants.SPROCKET_RADIUS
        qd_comp[1] = v_enc_r / constants.SPROCKET_RADIUS
        return qd_comp, beta_l, beta_r

    def _compensate_exp(self, v, omega, qd_des):
        # Exponential model compensation (same as original code)
        if abs(omega) < 1e-05 and abs(v) > 1e-05:
            radius = 1e08 * np.sign(v)
        elif abs(omega) < 1e-05 and abs(v) < 1e-05:
            radius = 1e8
        else:
            radius = v / omega

        v_enc_l = constants.SPROCKET_RADIUS * qd_des[0]
        v_enc_r = constants.SPROCKET_RADIUS * qd_des[1]

        if radius >= 0.0:
            beta_l = constants.beta_slip_inner_coefficients_left[0] * \
                     np.exp(constants.beta_slip_inner_coefficients_left[1] * radius)
            v_enc_l += beta_l
            beta_r = constants.beta_slip_outer_coefficients_left[0] * \
                     np.exp(constants.beta_slip_outer_coefficients_left[1] * radius)
            v_enc_r += beta_r
        else:
            beta_r = constants.beta_slip_inner_coefficients_right[0] * \
                     np.exp(constants.beta_slip_inner_coefficients_right[1] * radius)
            v_enc_r += beta_r
            beta_l = constants.beta_slip_outer_coefficients_right[0] * \
                     np.exp(constants.beta_slip_outer_coefficients_right[1] * radius)
            v_enc_l += beta_l

        qd_comp = np.zeros(2)
        qd_comp[0] = v_enc_l / constants.SPROCKET_RADIUS
        qd_comp[1] = v_enc_r / constants.SPROCKET_RADIUS
        return qd_comp, beta_l, beta_r
