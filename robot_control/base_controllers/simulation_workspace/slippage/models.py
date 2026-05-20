#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slip model loader with automatic detection of available models.
"""
import os
import glob
from termcolor import colored


class SlipModelLoader:
    """
    Scans for available slip prediction models and loads the correct one.
    """

    def __init__(self, sim):
        self.sim = sim

    def find_model_files(self, model_type='beta_l'):
        """
        Find available model files for a given type.

        Args:
            model_type: 'beta_l', 'beta_r', 'alpha'

        Returns:
            list of matching file paths
        """
        base_dir = os.environ.get('LOCOSIM_DIR', '') + \
                   '/robot_control/base_controllers/tracked_robot/regressor/'
        pattern = f'model_{model_type}*'
        files = glob.glob(os.path.join(base_dir, pattern))
        return files

    def list_available_models(self):
        """Print all available model files."""
        print(colored("Available slip models:", "cyan"))
        for t in ['beta_l', 'beta_r', 'alpha']:
            files = self.find_model_files(t)
            for f in files:
                print(f"  {os.path.basename(f)}")

    def extract_friction_from_filename(self, filename):
        """
        Extract friction coefficient from model filename.

        Example: 'model_beta_l_3d_0.6.cb' -> 0.6
        """
        import re
        match = re.search(r'(\d+\.?\d*)\.cb$', filename)
        if match:
            return float(match.group(1))
        return None