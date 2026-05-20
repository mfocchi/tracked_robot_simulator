#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground contact models for 2D and 3D simulations.
Re-exports the existing Ground classes.
"""

# Import existing ground classes
from base_controllers.open_loop_simulation2d import Ground
from base_controllers.open_loop_simulation3d import Ground3D

# Add any extensions if needed
class Ground2D(Ground):
    """
    Extended 2D ground model with additional parameters.
    """
    def __init__(self, friction_coefficient=0.4, **kwargs):
        super().__init__(friction_coefficient=friction_coefficient, **kwargs)
        self.type = '2D'

class Ground3DWithTerrain(Ground3D):
    """
    Extended 3D ground model that integrates with terrain.
    """
    def __init__(self, friction_coefficient=0.6, terrain_stiffness=1e05,
                 terrain_damping=0.5e04, **kwargs):
        super().__init__(
            friction_coefficient=friction_coefficient,
            terrain_stiffness=terrain_stiffness,
            terrain_damping=terrain_damping,
            **kwargs
        )
        self.type = '3D'

__all__ = ['Ground', 'Ground3D', 'Ground2D', 'Ground3DWithTerrain']