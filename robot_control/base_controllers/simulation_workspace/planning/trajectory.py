#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory module - re-exports existing Trajectory and ModelsList classes.
"""
# Re-export from the existing location
from base_controllers.tracked_robot.environment.trajectory import (
    Trajectory,
    ModelsList
)

# Add any additional trajectory utilities here if needed

__all__ = ['Trajectory', 'ModelsList']